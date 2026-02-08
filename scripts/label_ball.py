"""Ball labeling tool: extract frames with HSV candidates, then serve a web UI for annotation.

Workflow:
  1. prep   — extract ~500 frames from LRV files, run HSV detection, save JPEGs + prep.json
  2. serve  — Flask web UI where user clicks correct bounding box per frame
  3. train  — tune HSV params and train CNN classifier from annotations
  4. batch  — iterative batch labeling: label small batches, retrain, repeat

Usage:
  python scripts/label_ball.py prep [--count 500] [--force]
  python scripts/label_ball.py serve [--port 5001]
  python scripts/label_ball.py train
  python scripts/label_ball.py batch [--schedule 50,50,50,50,50,50]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABELING_DIR = Path("data/labeling")
FRAMES_DIR = LABELING_DIR / "frames"
PREP_PATH = LABELING_DIR / "prep.json"
ANNOTATIONS_PATH = LABELING_DIR / "annotations.json"
HSV_PARAMS_PATH = LABELING_DIR / "hsv_params.json"
BATCH_STATE_PATH = LABELING_DIR / "batch_state.json"
GAME_MASKS_PATH = LABELING_DIR / "game_masks.json"
MODEL_DIR = LABELING_DIR / "models"


def _migrate_v1_annotations(ann_data: dict) -> dict:
    """Migrate v1 annotation format to v2.

    v1: bare integers for candidate indices
    v2: {"type": "candidate", "index": N} or {"type": "click", "x": ..., "y": ...}
    """
    if ann_data.get("version", 1) >= 2:
        return ann_data
    frames = ann_data.get("frames", {})
    migrated = {}
    for name, value in frames.items():
        if isinstance(value, int):
            migrated[name] = {"type": "candidate", "index": value}
        else:
            migrated[name] = value  # null, "skip" pass through
    return {"version": 2, "frames": migrated}


def _load_annotations() -> dict:
    """Load annotations, migrating from v1 if needed."""
    if not ANNOTATIONS_PATH.exists():
        return {}
    ann_data = json.loads(ANNOTATIONS_PATH.read_text())
    ann_data = _migrate_v1_annotations(ann_data)
    # Re-save if we migrated
    if ann_data.get("version", 1) >= 2:
        ANNOTATIONS_PATH.write_text(json.dumps(ann_data, indent=2))
    return ann_data.get("frames", {})

def _nv12_to_bgr_bt709(nv12: np.ndarray, height: int) -> np.ndarray:
    """Convert NV12 buffer to BGR using BT.709 coefficients.

    OpenCV's COLOR_YUV2BGR_NV12 uses BT.601 which is wrong for HD/4K H.265
    content. BT.709 is the correct standard for HD video.
    """
    # NV12 layout: Y plane (height rows), UV plane (height/2 rows, interleaved U,V)
    y_plane = nv12[:height, :].astype(np.float32)
    uv_plane = nv12[height:, :]
    width = y_plane.shape[1]

    # Separate U and V (interleaved in NV12)
    u = uv_plane[:, 0::2].astype(np.float32) - 128.0
    v = uv_plane[:, 1::2].astype(np.float32) - 128.0

    # Upsample UV to full resolution (nearest neighbor, same as hardware)
    u = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)[:height, :width]
    v = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)[:height, :width]

    # BT.709 YUV → RGB
    r = y_plane + 1.5748 * v
    g = y_plane - 0.1873 * u - 0.4681 * v
    b = y_plane + 1.8556 * u

    bgr = np.stack([b, g, r], axis=-1)
    return np.clip(bgr, 0, 255).astype(np.uint8)


def _nvdec_read_frames(
    mp4_path: str | Path, frame_indices: list[int]
) -> dict[int, np.ndarray]:
    """Read specific frames from an MP4 using NVIDIA hardware decoder.

    Uses NVDEC seek for each target frame — much faster than cv2 for H.265.
    Returns dict mapping frame_index → BGR numpy array.
    Falls back to cv2.VideoCapture if NVDEC is not available.
    """
    try:
        import torch
        import PyNvVideoCodec as nvc
    except ImportError:
        # Fallback to cv2
        result = {}
        cap = cv2.VideoCapture(str(mp4_path))
        for idx in sorted(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                result[idx] = frame
        cap.release()
        return result

    mp4_str = str(mp4_path)
    result: dict[int, np.ndarray] = {}

    # Probe frame height from container (NV12 buffer is 1.5x this height)
    cap = cv2.VideoCapture(mp4_str)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    for target_frame in sorted(frame_indices):
        try:
            demuxer = nvc.CreateDemuxer(mp4_str)
            dec = nvc.CreateDecoder(gpuid=0, codec=demuxer.GetNvCodecId())
            ts = demuxer.TimestampFromFrame(target_frame)
            seek_packet = demuxer.Seek(ts)

            got_frame = False
            # Decode the seek packet
            frames = dec.Decode(seek_packet)
            for f in frames:
                nv12 = torch.from_dlpack(f).numpy().copy()
                bgr = _nv12_to_bgr_bt709(nv12, frame_height)
                result[target_frame] = bgr
                got_frame = True
                break

            if not got_frame:
                # Continue decoding until we get a frame
                for i, packet in enumerate(demuxer):
                    frames = dec.Decode(packet)
                    for f in frames:
                        nv12 = torch.from_dlpack(f).numpy().copy()
                        bgr = _nv12_to_bgr_bt709(nv12, frame_height)
                        result[target_frame] = bgr
                        got_frame = True
                        break
                    if got_frame or i > 100:
                        break
        except Exception:
            # Fallback for this frame
            cap = cv2.VideoCapture(mp4_str)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if ret:
                result[target_frame] = frame
            cap.release()

    return result


def _load_game_masks() -> dict[str, list[list[int]]]:
    """Load game area polygon masks from game_masks.json."""
    if not GAME_MASKS_PATH.exists():
        return {}
    return json.loads(GAME_MASKS_PATH.read_text())


def _make_game_area_mask(
    polygon: list[list[int]], height: int, width: int
) -> np.ndarray:
    """Create a binary mask from a polygon (list of [x, y] points)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask


# ---------------------------------------------------------------------------
# Prep command
# ---------------------------------------------------------------------------


BALL_REF_PATH = Path("data/ball_reference.webp")


def cmd_prep(args: argparse.Namespace) -> None:
    """Extract frames from LRV files, run HSV detection, write prep.json."""
    from wpv.tracking.detector import BallReferenceScorer, detect_hsv_candidates, detect_pool_mask

    # Safety check: don't clobber existing annotations
    if ANNOTATIONS_PATH.exists() and not args.force:
        print(
            f"ERROR: {ANNOTATIONS_PATH} already exists. Use --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Find all MP4 files (full resolution)
    mp4_files = sorted(Path("data").glob("PRO_VID_*/PRO_VID_*.mp4"))
    if not mp4_files:
        mp4_files = sorted(Path("data").glob("PRO_VID_*/PRO_VID_*.MP4"))
    if not mp4_files:
        print("ERROR: No MP4 files found matching data/PRO_VID_*/PRO_VID_*.mp4", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(mp4_files)} MP4 files")

    # Get frame counts for proportional sampling
    clip_info: list[tuple[Path, int]] = []
    total_frames = 0
    for mp4 in mp4_files:
        cap = cv2.VideoCapture(str(mp4))
        if not cap.isOpened():
            print(f"  WARNING: Cannot open {mp4}, skipping")
            continue
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n > 0:
            clip_info.append((mp4, n))
            total_frames += n
            print(f"  {mp4.name}: {n} frames")

    if total_frames == 0:
        print("ERROR: No readable frames in any MP4 file", file=sys.stderr)
        sys.exit(1)

    # Allocate frames proportionally
    target = args.count
    allocations: list[tuple[Path, int, list[int]]] = []
    allocated = 0
    for i, (lrv, n) in enumerate(clip_info):
        if i == len(clip_info) - 1:
            count = target - allocated  # last clip gets remainder
        else:
            count = max(1, round(target * n / total_frames))
        count = min(count, n)  # can't exceed clip length
        allocated += count

        # Uniform sampling within clip
        if count >= n:
            indices = list(range(n))
        else:
            indices = [round(j * (n - 1) / (count - 1)) for j in range(count)] if count > 1 else [n // 2]
        allocations.append((lrv, n, indices))

    print(f"\nSampling {allocated} frames across {len(clip_info)} clips")

    # Create output dirs
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # Extract frames and run HSV detection
    prep_data: dict = {
        "version": 1,
        "total_frames": allocated,
        "clips": [],
        "frames": [],
    }

    # Reference scorer — score candidates by similarity to ball picture
    ref_scorer = None
    if BALL_REF_PATH.exists():
        ref_scorer = BallReferenceScorer(BALL_REF_PATH)
        print(f"Using ball reference image: {BALL_REF_PATH}")
    else:
        print("WARNING: No ball reference image found, candidates will be unscored")

    frame_list: list[dict] = []
    clip_idx = 0
    game_masks = _load_game_masks()
    hsv_bands = _load_hsv_bands()
    for mp4, n_frames, indices in allocations:
        clip_name = f"clip_{clip_idx:03d}"
        prep_data["clips"].append({"clip_idx": clip_idx, "file": str(mp4), "total_frames": n_frames})
        print(f"  {clip_name}: reading {len(indices)} frames via NVDEC...", flush=True)

        # Read all target frames via NVDEC (hardware decode, ~54x faster than cv2 seeking)
        decoded_frames = _nvdec_read_frames(mp4, sorted(indices))
        if not decoded_frames:
            print(f"  WARNING: no frames decoded from {mp4}, skipping")
            clip_idx += 1
            continue

        # Compute pool mask once per clip from the first frame (camera is stationary)
        first_frame = decoded_frames.get(sorted(indices)[0])
        pool_mask = detect_pool_mask(first_frame) if first_frame is not None else None

        # Apply game area mask if available
        if pool_mask is not None and clip_name in game_masks:
            game_mask = _make_game_area_mask(
                game_masks[clip_name], pool_mask.shape[0], pool_mask.shape[1]
            )
            pool_mask = cv2.bitwise_and(pool_mask, game_mask)

        extracted = 0
        for frame_no in sorted(indices):
            frame = decoded_frames.get(frame_no)
            if frame is None:
                continue

            frame_name = f"{clip_name}_frame_{frame_no:06d}"
            jpeg_path = FRAMES_DIR / f"{frame_name}.jpg"

            # Save JPEG (quality 90)
            cv2.imwrite(str(jpeg_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # Run HSV detection (filtered to pool region)
            candidates = detect_hsv_candidates(frame, pool_mask=pool_mask, hsv_bands=hsv_bands)

            # Score candidates by reference ball similarity
            if ref_scorer and candidates:
                scored = ref_scorer.score(frame, candidates)
            else:
                scored = [(c, 0.5) for c in candidates]

            candidates_data = []
            for c, score in scored:
                candidates_data.append({
                    "bbox": list(c.bbox),
                    "centroid": list(c.centroid),
                    "area": c.area,
                    "circularity": c.circularity,
                    "mean_hsv": list(c.mean_hsv),
                    "ref_score": round(score, 3),
                })

            frame_list.append({
                "name": frame_name,
                "clip_idx": clip_idx,
                "frame_idx": frame_no,
                "file": str(mp4),
                "width": frame.shape[1],
                "height": frame.shape[0],
                "candidates": candidates_data,
            })

            extracted += 1
            if extracted % 10 == 0:
                print(f"    {extracted}/{len(indices)} processed", flush=True)

        print(f"  {clip_name} ({mp4.name}): {extracted} frames, "
              f"{sum(len(f['candidates']) for f in frame_list if f['clip_idx'] == clip_idx)} total candidates",
              flush=True)
        clip_idx += 1

    prep_data["frames"] = frame_list

    # Write prep.json
    PREP_PATH.write_text(json.dumps(prep_data, indent=2))
    print(f"\nWrote {len(frame_list)} frames to {PREP_PATH}")

    total_candidates = sum(len(f["candidates"]) for f in frame_list)
    high_conf = sum(
        1 for f in frame_list for c in f["candidates"] if c.get("ref_score", 0) > 0.6
    )
    frames_with_ball = sum(
        1 for f in frame_list
        if any(c.get("ref_score", 0) > 0.6 for c in f["candidates"])
    )
    print(f"\nTotal HSV candidates: {total_candidates} "
          f"(avg {total_candidates / max(1, len(frame_list)):.1f}/frame)")
    print(f"High-confidence candidates (score>0.6): {high_conf}")
    print(f"Frames with likely ball: {frames_with_ball}/{len(frame_list)}")


# ---------------------------------------------------------------------------
# Serve command — Flask web UI
# ---------------------------------------------------------------------------

MASK_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Game Area Mask — Clip {{ clip_idx }} ({{ clip_num }}/{{ total_clips }})</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #1a1a2e; color: #eee; font-family: system-ui, sans-serif;
         overflow: hidden; height: 100vh; display: flex; flex-direction: column; }
  .header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 8px 16px; background: #16213e; flex-shrink: 0;
  }
  .header h1 { font-size: 16px; font-weight: 500; }
  .progress-bar {
    width: 300px; height: 8px; background: #333; border-radius: 4px; overflow: hidden;
  }
  .progress-fill {
    height: 100%; background: #0f3460; border-radius: 4px;
    transition: width 0.3s;
  }
  .progress-text { font-size: 13px; color: #aaa; }
  .main { flex: 1; display: flex; flex-direction: column; align-items: center;
           padding: 8px; overflow: hidden; }
  .canvas-wrap { position: relative; cursor: crosshair; flex: 1;
                  display: flex; align-items: center; justify-content: center; }
  canvas { display: block; }
  #zoom-inset {
    position: absolute; width: 200px; height: 200px;
    border: 2px solid #e94560; border-radius: 4px;
    pointer-events: none; display: none;
    image-rendering: pixelated; overflow: hidden;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
  }
  #zoom-canvas { image-rendering: pixelated; }
  .controls {
    display: flex; gap: 8px; margin-top: 6px; flex-wrap: wrap; justify-content: center;
    flex-shrink: 0;
  }
  .controls button {
    padding: 6px 14px; border: 1px solid #555; border-radius: 4px;
    background: #16213e; color: #eee; cursor: pointer; font-size: 13px;
  }
  .controls button:hover { background: #0f3460; }
  .controls button.active { background: #e94560; border-color: #e94560; }
  .controls button.save-btn { background: #2a9d8f; border-color: #2a9d8f; }
  .controls button.save-btn:hover { background: #238b7e; }
  .info { font-size: 12px; color: #888; margin-top: 4px; flex-shrink: 0; }
  .toast {
    position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
    padding: 8px 20px; background: #0f3460; border-radius: 6px;
    font-size: 14px; opacity: 0; transition: opacity 0.3s;
    pointer-events: none; z-index: 100;
  }
  .toast.show { opacity: 1; }
</style>
</head>
<body>

<div class="header">
  <h1>Game Area Mask — Clip {{ clip_idx }}</h1>
  <div style="display:flex; align-items:center; gap:12px;">
    <a href="/masks" style="color:#0af;font-size:13px;text-decoration:none;">All Masks</a>
    <a href="/label/0" style="color:#0af;font-size:13px;text-decoration:none;">Labeling</a>
    <div class="progress-bar"><div class="progress-fill" id="progress-fill" style="width:{{ progress_pct }}%"></div></div>
    <span class="progress-text" id="progress-text">{{ clip_num }}/{{ total_clips }} clips</span>
  </div>
</div>

<div class="main">
  <div class="canvas-wrap" id="canvas-wrap">
    <canvas id="canvas"></canvas>
    <div id="zoom-inset"><canvas id="zoom-canvas" width="200" height="200"></canvas></div>
  </div>

  <div class="controls">
    <button onclick="navClip(-1)" id="prev-clip"><kbd>&larr;</kbd> Prev Clip</button>
    <button onclick="undoPoint()"><kbd>Z</kbd> Undo Point</button>
    <button onclick="clearPoly()"><kbd>C</kbd> Clear</button>
    <button id="cone-btn" onclick="toggleCones()"><kbd>H</kbd> Highlight Cones</button>
    <button class="save-btn" onclick="saveMask()"><kbd>Enter</kbd> Save & Next</button>
    <button onclick="navClip(1)" id="next-clip"><kbd>&rarr;</kbd> Next Clip</button>
  </div>

  <div class="info" id="info">{{ info_text }}</div>
</div>

<div class="toast" id="toast"></div>

<script>
const CLIP_IDX = {{ clip_idx }};
const CLIP_NAME = '{{ clip_name }}';
const EXISTING_POLY = {{ existing_poly_json | safe }};
const CONE_CANDIDATES = {{ cone_candidates_json | safe }};
const REVIEW_MODE = {{ review_mode }};
const TOTAL_CLIPS = {{ total_clips }};
const ZOOM_SIZE = 200;
const ZOOM_FACTOR = 8;

let points = EXISTING_POLY ? EXISTING_POLY.slice() : [];
let showCones = false;
let img = new Image();

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const zoomInset = document.getElementById('zoom-inset');
const zoomCanvas = document.getElementById('zoom-canvas');
const zoomCtx = zoomCanvas.getContext('2d');

img.onload = function() {
  // Use viewport size for canvas — fill available space
  let wrap = document.getElementById('canvas-wrap');
  let maxW = window.innerWidth - 24;
  let maxH = wrap.clientHeight - 4;
  let scale = Math.min(maxW / img.width, maxH / img.height, 1);
  canvas.width = Math.round(img.width * scale);
  canvas.height = Math.round(img.height * scale);
  canvas._scale = scale;
  canvas._imgW = img.width;
  canvas._imgH = img.height;
  draw();
};
img.src = '/api/mask-frame/' + CLIP_IDX;

function draw() {
  let s = canvas._scale;
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  // Draw cone candidate markers if toggled
  if (showCones && CONE_CANDIDATES) {
    CONE_CANDIDATES.forEach(function(c) {
      let cx = c[0] * s, cy = c[1] * s;
      let color = c[2] === 'yellow' ? 'rgba(255,255,0,0.8)' : 'rgba(255,50,50,0.8)';
      ctx.beginPath();
      ctx.arc(cx, cy, 8, 0, Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.stroke();
      // Outer glow
      ctx.beginPath();
      ctx.arc(cx, cy, 12, 0, Math.PI * 2);
      ctx.strokeStyle = color.replace('0.8', '0.3');
      ctx.lineWidth = 2;
      ctx.stroke();
    });
  }

  if (points.length === 0) { updateInfo(); return; }

  // Semi-transparent fill
  if (points.length >= 3) {
    ctx.beginPath();
    ctx.moveTo(points[0][0] * s, points[0][1] * s);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i][0] * s, points[i][1] * s);
    }
    ctx.closePath();
    ctx.fillStyle = 'rgba(42, 157, 143, 0.2)';
    ctx.fill();
  }

  // Lines between points
  ctx.beginPath();
  ctx.moveTo(points[0][0] * s, points[0][1] * s);
  for (let i = 1; i < points.length; i++) {
    ctx.lineTo(points[i][0] * s, points[i][1] * s);
  }
  if (points.length >= 3) ctx.closePath();
  ctx.strokeStyle = '#2a9d8f';
  ctx.lineWidth = 2;
  ctx.stroke();

  // Vertices
  points.forEach(function(p, i) {
    ctx.beginPath();
    ctx.arc(p[0] * s, p[1] * s, 6, 0, Math.PI * 2);
    ctx.fillStyle = i === points.length - 1 ? '#e94560' : '#2a9d8f';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Number label
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 13px system-ui';
    ctx.fillText('' + (i + 1), p[0] * s + 9, p[1] * s - 9);
  });

  updateInfo();
}

function updateInfo() {
  document.getElementById('info').textContent =
    points.length + ' points. Click on yellow cones to mark game area corners. Need >= 3 to save.';
}

function toggleCones() {
  showCones = !showCones;
  document.getElementById('cone-btn').classList.toggle('active', showCones);
  draw();
}

canvas.addEventListener('click', function(e) {
  let rect = canvas.getBoundingClientRect();
  let mx = e.clientX - rect.left;
  let my = e.clientY - rect.top;
  let s = canvas._scale;
  let ix = Math.round(mx / s);
  let iy = Math.round(my / s);

  // Snap to nearest cone candidate if within 30px (in screen coords)
  if (CONE_CANDIDATES) {
    let bestDist = 30;
    let snapX = ix, snapY = iy;
    CONE_CANDIDATES.forEach(function(c) {
      let dx = c[0] * s - mx;
      let dy = c[1] * s - my;
      let d = Math.sqrt(dx*dx + dy*dy);
      if (d < bestDist) {
        bestDist = d;
        snapX = c[0];
        snapY = c[1];
      }
    });
    ix = snapX;
    iy = snapY;
  }

  points.push([ix, iy]);
  draw();
});

// Zoom inset on hover
canvas.addEventListener('mousemove', function(e) {
  let rect = canvas.getBoundingClientRect();
  let mx = e.clientX - rect.left;
  let my = e.clientY - rect.top;
  let s = canvas._scale;

  zoomInset.style.display = 'block';
  let zx = mx + 25, zy = my - 100;
  if (zx + ZOOM_SIZE + 10 > canvas.width) zx = mx - ZOOM_SIZE - 25;
  if (zy < 0) zy = my + 25;
  zoomInset.style.left = zx + 'px';
  zoomInset.style.top = zy + 'px';

  let srcSize = ZOOM_SIZE / ZOOM_FACTOR;
  let srcX = mx / s - srcSize / 2;
  let srcY = my / s - srcSize / 2;
  zoomCtx.clearRect(0, 0, ZOOM_SIZE, ZOOM_SIZE);
  zoomCtx.drawImage(img, srcX, srcY, srcSize, srcSize, 0, 0, ZOOM_SIZE, ZOOM_SIZE);

  // Draw cone markers in zoom
  if (showCones && CONE_CANDIDATES) {
    CONE_CANDIDATES.forEach(function(c) {
      let rx = (c[0] - srcX) * ZOOM_FACTOR;
      let ry = (c[1] - srcY) * ZOOM_FACTOR;
      if (rx > -20 && rx < ZOOM_SIZE + 20 && ry > -20 && ry < ZOOM_SIZE + 20) {
        let color = c[2] === 'yellow' ? 'rgba(255,255,0,0.9)' : 'rgba(255,50,50,0.9)';
        zoomCtx.beginPath();
        zoomCtx.arc(rx, ry, 6, 0, Math.PI * 2);
        zoomCtx.strokeStyle = color;
        zoomCtx.lineWidth = 2;
        zoomCtx.stroke();
      }
    });
  }

  // Draw polygon edges in zoom
  if (points.length >= 2) {
    zoomCtx.beginPath();
    let p0 = points[0];
    zoomCtx.moveTo((p0[0] - srcX) * ZOOM_FACTOR, (p0[1] - srcY) * ZOOM_FACTOR);
    for (let i = 1; i < points.length; i++) {
      let p = points[i];
      zoomCtx.lineTo((p[0] - srcX) * ZOOM_FACTOR, (p[1] - srcY) * ZOOM_FACTOR);
    }
    if (points.length >= 3) zoomCtx.closePath();
    zoomCtx.strokeStyle = '#2a9d8f';
    zoomCtx.lineWidth = 2;
    zoomCtx.stroke();
  }

  // Crosshair
  let ch = ZOOM_SIZE / 2;
  zoomCtx.strokeStyle = 'rgba(255,255,255,0.7)';
  zoomCtx.lineWidth = 1;
  zoomCtx.beginPath(); zoomCtx.moveTo(ch, ch-12); zoomCtx.lineTo(ch, ch+12); zoomCtx.stroke();
  zoomCtx.beginPath(); zoomCtx.moveTo(ch-12, ch); zoomCtx.lineTo(ch+12, ch); zoomCtx.stroke();
});

canvas.addEventListener('mouseleave', function() {
  zoomInset.style.display = 'none';
});

function undoPoint() {
  if (points.length > 0) {
    points.pop();
    draw();
    showToast('Removed last point');
  }
}

function clearPoly() {
  points = [];
  draw();
  showToast('Cleared all points');
}

function saveMask() {
  if (points.length < 3) {
    showToast('Need at least 3 points!');
    return;
  }
  fetch('/api/save-mask', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({clip_name: CLIP_NAME, polygon: points, review: REVIEW_MODE})
  }).then(r => r.json()).then(function(data) {
    if (data.ok) {
      showToast('Saved! Advancing...');
      setTimeout(function() {
        if (data.next_clip !== null && data.next_clip !== undefined) {
          window.location.href = '/mask/' + data.next_clip;
        } else {
          window.location.href = '/';
        }
      }, 400);
    } else {
      showToast(data.error || 'Error saving');
    }
  });
}

function navClip(delta) {
  let next = CLIP_IDX + delta;
  if (next >= 0 && next < TOTAL_CLIPS) {
    window.location.href = '/mask/' + next;
  }
}

function showToast(msg) {
  let t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(function() { t.classList.remove('show'); }, 1500);
}

// Hide prev/next buttons at boundaries
if (CLIP_IDX <= 0) document.getElementById('prev-clip').style.display = 'none';
if (CLIP_IDX >= TOTAL_CLIPS - 1) document.getElementById('next-clip').style.display = 'none';

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
  if (e.key === 'ArrowLeft') { navClip(-1); e.preventDefault(); return; }
  if (e.key === 'ArrowRight') { navClip(1); e.preventDefault(); return; }
  if (e.key === 'z' || e.key === 'Z') { undoPoint(); e.preventDefault(); }
  else if (e.key === 'c' || e.key === 'C') { clearPoly(); e.preventDefault(); }
  else if (e.key === 'h' || e.key === 'H') { toggleCones(); e.preventDefault(); }
  else if (e.key === 'Enter') { saveMask(); e.preventDefault(); }
});
</script>
</body>
</html>
"""

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Ball Labeling — {{ frame_name }} ({{ idx+1 }}/{{ total }})</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #1a1a2e; color: #eee; font-family: system-ui, sans-serif; }
  .header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 8px 16px; background: #16213e;
  }
  .header h1 { font-size: 16px; font-weight: 500; }
  .progress-bar {
    width: 300px; height: 8px; background: #333; border-radius: 4px; overflow: hidden;
  }
  .progress-fill {
    height: 100%; background: #0f3460; border-radius: 4px;
    transition: width 0.3s;
  }
  .progress-text { font-size: 13px; color: #aaa; }
  .main { display: flex; flex-direction: column; align-items: center; padding: 12px; }
  .canvas-wrap { position: relative; cursor: crosshair; }
  canvas { display: block; }
  #zoom-inset {
    position: absolute; width: 160px; height: 160px;
    border: 2px solid #e94560; border-radius: 4px;
    pointer-events: none; display: none;
    image-rendering: pixelated; overflow: hidden;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
  }
  #zoom-canvas { image-rendering: pixelated; }
  .controls {
    display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; justify-content: center;
  }
  .controls button {
    padding: 6px 14px; border: 1px solid #555; border-radius: 4px;
    background: #16213e; color: #eee; cursor: pointer; font-size: 13px;
  }
  .controls button:hover { background: #0f3460; }
  .controls button.active { background: #e94560; border-color: #e94560; }
  .controls button kbd {
    display: inline-block; padding: 1px 4px; margin-right: 4px;
    background: rgba(255,255,255,0.15); border-radius: 2px; font-size: 11px;
  }
  .nav { display: flex; gap: 8px; margin-top: 8px; }
  .nav button { padding: 6px 18px; }
  .info { font-size: 12px; color: #888; margin-top: 8px; }
  .candidate-list {
    margin-top: 8px; display: flex; gap: 6px; flex-wrap: wrap; justify-content: center;
  }
  .candidate-chip {
    padding: 3px 10px; border-radius: 12px; font-size: 12px;
    background: #16213e; border: 1px solid #555; cursor: pointer;
  }
  .candidate-chip:hover { background: #0f3460; }
  .candidate-chip.selected { background: #e94560; border-color: #e94560; }
  .toast {
    position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
    padding: 8px 20px; background: #0f3460; border-radius: 6px;
    font-size: 14px; opacity: 0; transition: opacity 0.3s;
    pointer-events: none; z-index: 100;
  }
  .toast.show { opacity: 1; }
</style>
</head>
<body>

<div class="header">
  <h1>Ball Labeling</h1>
  <div style="display:flex; align-items:center; gap:12px;">
    <a href="/masks" style="color:#0af;font-size:13px;text-decoration:none;">Review Masks</a>
    <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
    <span class="progress-text" id="progress-text">0/0</span>
  </div>
</div>

<div class="main">
  <div class="canvas-wrap" id="canvas-wrap">
    <canvas id="canvas"></canvas>
    <div id="zoom-inset"><canvas id="zoom-canvas" width="160" height="160"></canvas></div>
  </div>

  <div class="candidate-list" id="candidate-list"></div>

  <div class="controls">
    <button onclick="annotate(null)"><kbd>N</kbd> No Ball</button>
    <button onclick="annotate('skip')"><kbd>S</kbd> Skip</button>
    <button onclick="undo()"><kbd>U</kbd> Undo</button>
  </div>

  <div class="nav">
    <button onclick="navigate(-1)"><kbd>←</kbd> Prev</button>
    <button onclick="navigate(1)"><kbd>→</kbd> Next</button>
    <button onclick="goNextUnlabeled()"><kbd>Space</kbd> Next Unlabeled</button>
    <button onclick="finishBatch()" id="done-btn" style="background:#2a9d8f;border-color:#2a9d8f;"><kbd>D</kbd> Done</button>
  </div>

  <div class="info" id="info"></div>
</div>

<div class="toast" id="toast"></div>

<script>
const FRAME_DATA = {{ frame_data_json | safe }};
const TOTAL = {{ total }};
let currentIdx = {{ idx }};
let annotations = {{ annotations_json | safe }};
let img = new Image();
let candidates = [];
const HIT_PAD = 15;
const ZOOM_FACTOR = 6;

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const zoomInset = document.getElementById('zoom-inset');
const zoomCanvas = document.getElementById('zoom-canvas');
const zoomCtx = zoomCanvas.getContext('2d');

// Colors for candidate boxes
const COLORS = [
  '#e94560','#0f3460','#00b4d8','#e9c46a','#2a9d8f',
  '#f4845f','#a855f7','#22d3ee','#fb923c'
];

function loadFrame(idx) {
  currentIdx = idx;
  const fd = FRAME_DATA[idx];
  candidates = fd.candidates;
  img = new Image();
  img.onload = function() {
    // Scale to fit viewport
    let maxW = window.innerWidth - 40;
    let maxH = window.innerHeight - 200;
    let scale = Math.min(maxW / img.width, maxH / img.height, 1);
    canvas.width = Math.round(img.width * scale);
    canvas.height = Math.round(img.height * scale);
    canvas._scale = scale;
    drawFrame();
    updateUI();
  };
  img.src = '/api/frame/' + fd.name + '.jpg';
}

function drawFrame() {
  let s = canvas._scale;
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  // Draw candidate boxes
  candidates.forEach(function(c, i) {
    let [x, y, w, h] = c.bbox;
    x *= s; y *= s; w *= s; h *= s;
    let color = COLORS[i % COLORS.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);

    // Number label
    ctx.fillStyle = color;
    ctx.font = 'bold 14px system-ui';
    let label = '' + (i + 1);
    let tw = ctx.measureText(label).width;
    ctx.fillRect(x - 1, y - 16, tw + 6, 16);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x + 2, y - 3);
  });

  // Highlight current annotation if any
  let fd = FRAME_DATA[currentIdx];
  let ann = annotations[fd.name];
  if (ann && typeof ann === 'object' && ann.type === 'candidate') {
    let c = candidates[ann.index];
    if (c) {
      let [x, y, w, h] = c.bbox;
      x *= s; y *= s; w *= s; h *= s;
      ctx.strokeStyle = '#0f0';
      ctx.lineWidth = 3;
      ctx.strokeRect(x - 2, y - 2, w + 4, h + 4);
    }
  }
}

function updateUI() {
  let fd = FRAME_DATA[currentIdx];
  // Progress
  let labeled = Object.keys(annotations).length;
  document.getElementById('progress-fill').style.width = (100 * labeled / TOTAL) + '%';
  document.getElementById('progress-text').textContent = labeled + '/' + TOTAL + ' labeled';

  // Info
  let ann = annotations[fd.name];
  let status = 'unlabeled';
  if (ann === null) status = 'no ball';
  else if (ann === 'skip') status = 'skipped';
  else if (ann && typeof ann === 'object' && ann.type === 'candidate') status = 'box ' + (ann.index + 1);
  document.getElementById('info').textContent =
    fd.name + ' | ' + candidates.length + ' candidates | ' + status +
    ' | Frame ' + (currentIdx + 1) + '/' + TOTAL;

  // Candidate chips
  let list = document.getElementById('candidate-list');
  list.innerHTML = '';
  candidates.forEach(function(c, i) {
    let isSelected = ann && typeof ann === 'object' && ann.type === 'candidate' && ann.index === i;
    let chip = document.createElement('span');
    chip.className = 'candidate-chip' + (isSelected ? ' selected' : '');
    let label = (i + 1) + ' (' + Math.round(c.area) + 'px)';
    if (c.score !== undefined && c.score !== null) label += ' ' + (c.score * 100).toFixed(0) + '%';
    chip.textContent = label;
    chip.style.borderColor = COLORS[i % COLORS.length];
    chip.onclick = function() { annotate({"type": "candidate", "index": i}); };
    chip.onmouseenter = function() { showZoomAtCandidate(i); };
    chip.onmouseleave = hideZoom;
    list.appendChild(chip);
  });

  // Update URL without reload
  history.replaceState(null, '', '/label/' + currentIdx);
}

function annotate(value) {
  let fd = FRAME_DATA[currentIdx];
  fetch('/api/annotate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({frame: fd.name, value: value})
  }).then(r => r.json()).then(function(data) {
    annotations = data.annotations;
    let msg = 'Annotated';
    if (value === null) msg = 'No ball';
    else if (value === 'skip') msg = 'Skipped';
    else if (value && value.type === 'candidate') msg = 'Selected box ' + (value.index + 1);
    else if (value && value.type === 'click') msg = 'Clicked at (' + value.x + ',' + value.y + ')';
    showToast(msg);
    drawFrame();
    updateUI();
    // Auto-advance to next unlabeled
    setTimeout(function() { goNextUnlabeled(); }, 200);
  });
}

function undo() {
  fetch('/api/undo', {method: 'POST'})
    .then(r => r.json())
    .then(function(data) {
      if (data.undone) {
        annotations = data.annotations;
        // Navigate to the undone frame
        let undoneIdx = FRAME_DATA.findIndex(f => f.name === data.undone);
        if (undoneIdx >= 0) {
          loadFrame(undoneIdx);
        }
        showToast('Undone: ' + data.undone);
      } else {
        showToast('Nothing to undo');
      }
    });
}

function navigate(delta) {
  let next = currentIdx + delta;
  if (next >= 0 && next < TOTAL) loadFrame(next);
}

function goNextUnlabeled() {
  // Search forward from current, then wrap around
  for (let i = 1; i <= TOTAL; i++) {
    let idx = (currentIdx + i) % TOTAL;
    if (annotations[FRAME_DATA[idx].name] === undefined) {
      loadFrame(idx);
      return;
    }
  }
  showToast('All frames labeled!');
  drawFrame();
  updateUI();
}

function showZoomAtCandidate(ci) {
  let c = candidates[ci];
  if (!c) return;
  let s = canvas._scale;
  // Center zoom on candidate centroid
  let cx = c.centroid[0], cy = c.centroid[1];
  let srcSize = 160 / ZOOM_FACTOR;
  let srcX = cx - srcSize / 2;
  let srcY = cy - srcSize / 2;
  zoomCtx.clearRect(0, 0, 160, 160);
  zoomCtx.drawImage(img, srcX, srcY, srcSize, srcSize, 0, 0, 160, 160);
  // Draw candidate box in zoom
  let [bx, by, bw, bh] = c.bbox;
  zoomCtx.strokeStyle = COLORS[ci % COLORS.length];
  zoomCtx.lineWidth = 2;
  zoomCtx.strokeRect((bx - srcX) * ZOOM_FACTOR, (by - srcY) * ZOOM_FACTOR, bw * ZOOM_FACTOR, bh * ZOOM_FACTOR);
  // Crosshair
  zoomCtx.strokeStyle = 'rgba(255,255,255,0.7)';
  zoomCtx.lineWidth = 1;
  zoomCtx.beginPath(); zoomCtx.moveTo(80, 70); zoomCtx.lineTo(80, 90); zoomCtx.stroke();
  zoomCtx.beginPath(); zoomCtx.moveTo(70, 80); zoomCtx.lineTo(90, 80); zoomCtx.stroke();
  // Position zoom inset near the candidate on canvas
  let screenX = cx * s, screenY = cy * s;
  let zx = screenX + 20, zy = screenY - 80;
  if (zx + 170 > canvas.width) zx = screenX - 180;
  if (zy < 0) zy = screenY + 20;
  zoomInset.style.left = zx + 'px';
  zoomInset.style.top = zy + 'px';
  zoomInset.style.display = 'block';
}

function hideZoom() {
  zoomInset.style.display = 'none';
}

function finishBatch() {
  let labeled = Object.keys(annotations).filter(k => FRAME_DATA.some(f => f.name === k)).length;
  let total = FRAME_DATA.length;
  let msg = labeled < total
    ? 'Finish with ' + labeled + '/' + total + ' labeled?'
    : 'All ' + total + ' frames labeled. Finish batch?';
  if (!confirm(msg)) return;
  fetch('/api/finish-batch', {method: 'POST'})
    .then(r => r.json())
    .then(function(data) {
      if (!data.ok) { showToast(data.error || 'Error'); return; }
      showTrainingOverlay();
      pollBatchStatus();
    });
}

function showTrainingOverlay() {
  let overlay = document.createElement('div');
  overlay.id = 'training-overlay';
  overlay.style.cssText = 'position:fixed;inset:0;background:rgba(26,26,46,0.95);display:flex;flex-direction:column;align-items:center;justify-content:center;z-index:200;';
  overlay.innerHTML = '<div style="font-size:20px;margin-bottom:16px;">Training in progress...</div>'
    + '<div id="train-status" style="font-size:14px;color:#aaa;max-width:500px;text-align:center;"></div>'
    + '<div style="margin-top:24px;width:200px;height:4px;background:#333;border-radius:2px;overflow:hidden;">'
    + '<div id="train-bar" style="height:100%;width:20%;background:#0f3460;animation:pulse 1.5s ease-in-out infinite;"></div></div>';
  let style = document.createElement('style');
  style.textContent = '@keyframes pulse{0%,100%{transform:translateX(-100%)}50%{transform:translateX(400%)}}';
  overlay.appendChild(style);
  document.body.appendChild(overlay);
}

function pollBatchStatus() {
  fetch('/api/batch-status')
    .then(r => r.json())
    .then(function(data) {
      let el = document.getElementById('train-status');
      if (el) el.textContent = data.train_progress || '';
      if (data.phase === 'labeling') {
        // Next batch is ready — reload
        window.location.href = '/label/0';
      } else if (data.phase === 'done') {
        let overlay = document.getElementById('training-overlay');
        if (overlay) overlay.innerHTML = '<div style="font-size:20px;">All batches complete!</div>'
          + '<div style="font-size:14px;color:#aaa;margin-top:12px;">' + data.total_annotations + ' total annotations</div>';
      } else {
        setTimeout(pollBatchStatus, 1000);
      }
    })
    .catch(function() { setTimeout(pollBatchStatus, 2000); });
}

function showToast(msg) {
  let t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(function() { t.classList.remove('show'); }, 1500);
}

// Click on canvas — find nearest candidate or click-anywhere
canvas.addEventListener('click', function(e) {
  let rect = canvas.getBoundingClientRect();
  let mx = e.clientX - rect.left;
  let my = e.clientY - rect.top;
  let s = canvas._scale;

  // Check each candidate with expanded hit area
  let best = -1;
  let bestDist = Infinity;
  candidates.forEach(function(c, i) {
    let [x, y, w, h] = c.bbox;
    x *= s; y *= s; w *= s; h *= s;
    // Expand hit area
    let x0 = x - HIT_PAD, y0 = y - HIT_PAD;
    let x1 = x + w + HIT_PAD, y1 = y + h + HIT_PAD;
    if (mx >= x0 && mx <= x1 && my >= y0 && my <= y1) {
      let cx = x + w/2, cy = y + h/2;
      let d = Math.hypot(mx - cx, my - cy);
      if (d < bestDist) { bestDist = d; best = i; }
    }
  });

  if (best >= 0) {
    annotate({"type": "candidate", "index": best});
  }
});

// Zoom inset on hover — always on
canvas.addEventListener('mousemove', function(e) {
  let rect = canvas.getBoundingClientRect();
  let mx = e.clientX - rect.left;
  let my = e.clientY - rect.top;
  let s = canvas._scale;

  zoomInset.style.display = 'block';
  // Position zoom inset near cursor but not off-screen
  let zx = mx + 20, zy = my - 80;
  if (zx + 170 > canvas.width) zx = mx - 180;
  if (zy < 0) zy = my + 20;
  zoomInset.style.left = zx + 'px';
  zoomInset.style.top = zy + 'px';

  // Draw zoomed region (ZOOM_FACTOR=6, so 160/6 ≈ 26.7px source region)
  let srcSize = 160 / ZOOM_FACTOR;
  let srcX = mx / s - srcSize / 2;
  let srcY = my / s - srcSize / 2;
  zoomCtx.clearRect(0, 0, 160, 160);
  zoomCtx.drawImage(img, srcX, srcY, srcSize, srcSize, 0, 0, 160, 160);

  // Draw candidate boxes in zoom view
  candidates.forEach(function(c, i) {
    let [bx, by, bw, bh] = c.bbox;
    let rx = (bx - srcX) * ZOOM_FACTOR;
    let ry = (by - srcY) * ZOOM_FACTOR;
    let rw = bw * ZOOM_FACTOR;
    let rh = bh * ZOOM_FACTOR;
    zoomCtx.strokeStyle = COLORS[i % COLORS.length];
    zoomCtx.lineWidth = 2;
    zoomCtx.strokeRect(rx, ry, rw, rh);
  });

  // Crosshair at zoom center for precise clicking
  zoomCtx.strokeStyle = 'rgba(255,255,255,0.7)';
  zoomCtx.lineWidth = 1;
  zoomCtx.beginPath(); zoomCtx.moveTo(80, 70); zoomCtx.lineTo(80, 90); zoomCtx.stroke();
  zoomCtx.beginPath(); zoomCtx.moveTo(70, 80); zoomCtx.lineTo(90, 80); zoomCtx.stroke();
});

canvas.addEventListener('mouseleave', function() {
  zoomInset.style.display = 'none';
});

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
  let key = e.key;
  if (key >= '1' && key <= '9') {
    let idx = parseInt(key) - 1;
    if (idx < candidates.length) annotate({"type": "candidate", "index": idx});
    e.preventDefault();
  } else if (key === 'n' || key === 'N') {
    annotate(null);
    e.preventDefault();
  } else if (key === 's' || key === 'S') {
    annotate('skip');
    e.preventDefault();
  } else if (key === 'u' || key === 'U') {
    undo();
    e.preventDefault();
  } else if (key === 'd' || key === 'D') {
    finishBatch();
    e.preventDefault();
  } else if (key === 'ArrowLeft') {
    navigate(-1);
    e.preventDefault();
  } else if (key === 'ArrowRight') {
    navigate(1);
    e.preventDefault();
  } else if (key === ' ') {
    goNextUnlabeled();
    e.preventDefault();
  }
});

// Initial load
loadFrame(currentIdx);
</script>
</body>
</html>
"""


def cmd_serve(args: argparse.Namespace) -> None:
    """Launch Flask labeling UI."""
    try:
        from flask import Flask, jsonify, redirect, request
    except ImportError:
        print("ERROR: Flask not installed. Run: pip install flask", file=sys.stderr)
        sys.exit(1)

    if not PREP_PATH.exists():
        print(f"ERROR: {PREP_PATH} not found. Run prep first.", file=sys.stderr)
        sys.exit(1)

    prep = json.loads(PREP_PATH.read_text())
    frames = prep["frames"]
    if not frames:
        print("ERROR: No frames in prep.json", file=sys.stderr)
        sys.exit(1)

    # Load or create annotations (v2 format)
    annotations: dict = _load_annotations()

    # Undo stack
    undo_stack: list[tuple[str, object]] = []  # (frame_name, previous_value_or_sentinel)
    _SENTINEL = object()
    MAX_UNDO = 20

    def save_annotations() -> None:
        ANNOTATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        ANNOTATIONS_PATH.write_text(json.dumps({
            "version": 2,
            "frames": annotations,
        }, indent=2))

    app = Flask(__name__)

    @app.route("/")
    def index():
        return redirect("/label/0")

    @app.route("/label/<int:idx>")
    def label_page(idx: int):
        idx = max(0, min(idx, len(frames) - 1))
        # Minimal frame data for JS (just what's needed)
        frame_data = []
        for f in frames:
            frame_data.append({
                "name": f["name"],
                "candidates": f["candidates"],
            })

        # Render inline template
        html = HTML_TEMPLATE
        html = html.replace("{{ frame_data_json | safe }}", json.dumps(frame_data))
        html = html.replace("{{ total }}", str(len(frames)))
        html = html.replace("{{ idx }}", str(idx))
        html = html.replace("{{ idx+1 }}", str(idx + 1))
        html = html.replace("{{ frame_name }}", frames[idx]["name"])
        html = html.replace("{{ annotations_json | safe }}", json.dumps(annotations))
        return html

    @app.route("/api/frame/<name>")
    def serve_frame(name: str):
        from flask import send_file
        safe = Path(name).name
        path = FRAMES_DIR / safe
        if not path.exists():
            return "Not found", 404
        return send_file(path.resolve(), mimetype="image/jpeg")

    @app.route("/api/annotate", methods=["POST"])
    def api_annotate():
        data = request.get_json()
        frame_name = data["frame"]
        value = data["value"]

        # Push to undo stack
        prev = annotations.get(frame_name, _SENTINEL)
        undo_stack.append((frame_name, prev))
        if len(undo_stack) > MAX_UNDO:
            undo_stack.pop(0)

        annotations[frame_name] = value
        save_annotations()
        return jsonify({"ok": True, "annotations": annotations})

    @app.route("/api/undo", methods=["POST"])
    def api_undo():
        if not undo_stack:
            return jsonify({"undone": None, "annotations": annotations})
        frame_name, prev = undo_stack.pop()
        if prev is _SENTINEL:
            annotations.pop(frame_name, None)
        else:
            annotations[frame_name] = prev
        save_annotations()
        return jsonify({"undone": frame_name, "annotations": annotations})

    @app.route("/api/progress")
    def api_progress():
        return jsonify({
            "total": len(frames),
            "labeled": len(annotations),
            "remaining": len(frames) - len(annotations),
        })

    @app.route("/api/shutdown", methods=["POST"])
    def api_shutdown():
        save_annotations()
        func = request.environ.get("werkzeug.server.shutdown")
        if func:
            func()
        else:
            import signal
            import threading
            threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
        return jsonify({"ok": True})

    print(f"Labeling UI: http://0.0.0.0:{args.port}/")
    print(f"  {len(frames)} frames, {len(annotations)} already labeled")
    print("  Shortcuts: 1-9=select box, n=no ball, s=skip, u=undo, d=done, arrows=navigate, space=next unlabeled")
    app.run(host="0.0.0.0", port=args.port, debug=False)


# ---------------------------------------------------------------------------
# Train command
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> None:
    """Tune HSV params and train CNN from current annotations."""
    from wpv.tracking.detector import (
        HSV_BANDS,
        train_classifier,
        tune_hsv_from_labels,
    )

    if not PREP_PATH.exists():
        print(f"ERROR: {PREP_PATH} not found. Run prep first.", file=sys.stderr)
        sys.exit(1)

    annotations = _load_annotations()
    if not annotations:
        print("ERROR: No annotations found.", file=sys.stderr)
        sys.exit(1)

    prep_data = json.loads(PREP_PATH.read_text())

    # Load current HSV bands (from tuned params or defaults)
    if HSV_PARAMS_PATH.exists():
        hsv_data = json.loads(HSV_PARAMS_PATH.read_text())
        current_bands = [
            (tuple(b["lo"]), tuple(b["hi"])) for b in hsv_data["bands"]
        ]
    else:
        current_bands = list(HSV_BANDS)

    # Tune HSV
    print("Tuning HSV bands from labels...")
    new_bands = tune_hsv_from_labels(FRAMES_DIR, annotations, prep_data, current_bands)
    HSV_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    HSV_PARAMS_PATH.write_text(json.dumps({
        "bands": [{"lo": list(lo), "hi": list(hi)} for lo, hi in new_bands],
    }, indent=2))
    print(f"  Saved {len(new_bands)} HSV bands to {HSV_PARAMS_PATH}")
    for lo, hi in new_bands:
        print(f"    H:{lo[0]}-{hi[0]}  S:{lo[1]}-{hi[1]}  V:{lo[2]}-{hi[2]}")

    # Train CNN
    print("Training CNN classifier...")
    model_path = MODEL_DIR / "ball_classifier.pth"
    pretrained = model_path if model_path.exists() else None
    result = train_classifier(
        FRAMES_DIR, annotations, prep_data, model_path, pretrained_path=pretrained
    )
    if result.get("skipped"):
        print(f"  Skipped: {result['reason']} ({result['n_pos']} positives)")
    else:
        print(f"  Trained: {result['n_pos']} pos, {result['n_neg']} neg")
        print(f"  Best loss: {result['best_loss']:.4f}, Accuracy: {result['final_accuracy']:.1%}")
        print(f"  Model saved to {result['model_path']}")


# ---------------------------------------------------------------------------
# Batch command
# ---------------------------------------------------------------------------


def _load_hsv_bands():
    """Load tuned HSV bands or return None for defaults."""
    from wpv.tracking.detector import HSV_BANDS

    if HSV_PARAMS_PATH.exists():
        hsv_data = json.loads(HSV_PARAMS_PATH.read_text())
        return [
            (tuple(b["lo"]), tuple(b["hi"])) for b in hsv_data["bands"]
        ]
    return list(HSV_BANDS)


def _rerun_hsv_for_frames(frame_names: list[str], prep_data: dict) -> dict:
    """Re-run HSV detection on specific frames with latest tuned bands + CNN scoring."""
    from wpv.tracking.detector import BallClassifier, detect_hsv_candidates, detect_pool_mask

    hsv_bands = _load_hsv_bands()
    prep_lookup = {f["name"]: f for f in prep_data["frames"]}
    game_masks = _load_game_masks()

    # Load CNN if available
    model_path = MODEL_DIR / "ball_classifier.pth"
    classifier = BallClassifier(model_path if model_path.exists() else None)

    # Cache game area masks per clip (avoid recomputing per frame)
    game_area_cache: dict[str, np.ndarray | None] = {}

    updated = 0
    for name in frame_names:
        pf = prep_lookup.get(name)
        if pf is None:
            continue
        jpeg_path = FRAMES_DIR / f"{name}.jpg"
        if not jpeg_path.exists():
            continue
        img = cv2.imread(str(jpeg_path))
        if img is None:
            continue

        pool_mask = detect_pool_mask(img)

        # Apply game area mask if available
        clip_name = f"clip_{pf['clip_idx']:03d}"
        if clip_name not in game_area_cache:
            if clip_name in game_masks and pool_mask is not None:
                game_area_cache[clip_name] = _make_game_area_mask(
                    game_masks[clip_name], img.shape[0], img.shape[1]
                )
            else:
                game_area_cache[clip_name] = None
        if game_area_cache[clip_name] is not None and pool_mask is not None:
            pool_mask = cv2.bitwise_and(pool_mask, game_area_cache[clip_name])

        candidates = detect_hsv_candidates(img, pool_mask=pool_mask, hsv_bands=hsv_bands)

        # Score with CNN
        scored = classifier.predict(img, candidates)

        candidates_data = []
        for c, score in scored:
            candidates_data.append({
                "bbox": list(c.bbox),
                "centroid": list(c.centroid),
                "area": c.area,
                "circularity": c.circularity,
                "mean_hsv": list(c.mean_hsv),
                "score": round(score, 3),
            })
        # Sort by score descending so best candidate is first
        candidates_data.sort(key=lambda x: -x.get("score", 0))
        pf["candidates"] = candidates_data
        updated += 1

    # Write updated prep.json
    PREP_PATH.write_text(json.dumps(prep_data, indent=2))
    return prep_data


def _select_batch_frames(
    batch_idx: int,
    batch_size: int,
    prep_data: dict,
    annotations: dict,
    already_selected: set[str],
) -> list[str]:
    """Select frames for a batch, ensuring diversity across clips.

    Round-robin across clips so each batch covers varied lighting/conditions.
    Later batches also prioritize frames with 0 or many candidates.
    """
    available = [f for f in prep_data["frames"] if f["name"] not in already_selected]
    if not available:
        return []

    # Only show frames that have candidates — no point showing empty frames
    available = [f for f in available if len(f.get("candidates", [])) > 0]
    if not available:
        return []

    # Group by clip
    by_clip: dict[int, list[dict]] = {}
    for f in available:
        by_clip.setdefault(f["clip_idx"], []).append(f)

    if batch_idx > 0:
        # Within each clip, prefer frames with fewer candidates (1-3 = easier to label)
        for clip_frames in by_clip.values():
            clip_frames.sort(key=lambda f: len(f.get("candidates", [])))

    # Round-robin across clips
    selected: list[str] = []
    clip_ids = sorted(by_clip.keys())
    clip_iters = {c: iter(frames) for c, frames in by_clip.items()}
    while len(selected) < batch_size:
        picked_any = False
        for c in clip_ids:
            if len(selected) >= batch_size:
                break
            f = next(clip_iters[c], None)
            if f is not None:
                selected.append(f["name"])
                picked_any = True
        if not picked_any:
            break

    return selected


def _calibrate_from_saved_masks(
    prep_data: dict,
) -> tuple[float | None, list[list[int]] | None]:
    """Learn offset ratio and average polygon from user-corrected masks.

    For each saved mask, finds the detected yellow cones on that clip's first
    frame, pairs each boundary point to the nearest yellow, and computes
    distance(yellow→boundary) / sqrt(yellow_area).

    Returns:
      (median_ratio, avg_polygon):
        median_ratio — calibrated offset multiplier (replaces hardcoded 2.0),
                       or None if no saved masks
        avg_polygon  — element-wise mean polygon from all saved masks
                       (fallback for clips where detection fails entirely),
                       or None if no saved masks
    """
    from wpv.tracking.detector import detect_pool_mask

    game_masks = _load_game_masks()
    if not game_masks:
        return None, None

    clip_first_frame: dict[str, str] = {}
    for f in prep_data["frames"]:
        cname = f"clip_{f['clip_idx']:03d}"
        if cname not in clip_first_frame:
            clip_first_frame[cname] = f["name"]

    ratios: list[float] = []
    all_polys: list[list[list[int]]] = []

    for clip_name, saved_poly in game_masks.items():
        if len(saved_poly) < 3:
            continue
        all_polys.append(saved_poly)

        frame_name = clip_first_frame.get(clip_name)
        if not frame_name:
            continue
        frame_path = FRAMES_DIR / f"{frame_name}.jpg"
        if not frame_path.exists():
            continue
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        pool_mask = detect_pool_mask(img)
        if pool_mask is None:
            continue

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Pool geometry for long axis
        M = cv2.moments(pool_mask)
        if M["m00"] == 0:
            continue
        pcx = int(M["m10"] / M["m00"])
        pcy = int(M["m01"] / M["m00"])
        cts, _ = cv2.findContours(
            pool_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        pool_ct = max(cts, key=cv2.contourArea)
        rect = cv2.minAreaRect(pool_ct)
        rw, rh = rect[1]
        angle = rect[2]
        long_angle = np.radians(angle + 90) if rw < rh else np.radians(angle)
        long_dir = np.array([np.cos(long_angle), np.sin(long_angle)])

        # Find yellow cones (same logic as _detect_cone_candidates)
        k_out = np.ones((80, 80), np.uint8)
        k_in = np.ones((60, 60), np.uint8)
        dilated = cv2.dilate(pool_mask, k_out)
        eroded = cv2.erode(pool_mask, k_in)
        band = cv2.subtract(dilated, eroded)
        yellow_mask = cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))
        yellow_near = cv2.bitwise_and(yellow_mask, band)
        yellow_near = cv2.morphologyEx(
            yellow_near, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        contours, _ = cv2.findContours(
            yellow_near, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        yellows: list[tuple[int, int, float]] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 15:
                continue
            m = cv2.moments(c)
            if m["m00"] > 0:
                yx = int(m["m10"] / m["m00"])
                yy = int(m["m01"] / m["m00"])
                # Cluster
                if not any(
                    np.sqrt((yx - ex) ** 2 + (yy - ey) ** 2) < 80
                    for ex, ey, _ in yellows
                ):
                    yellows.append((yx, yy, area))

        if not yellows:
            continue

        # For each boundary point, find nearest yellow and compute ratio
        for bx, by in saved_poly:
            best_dist = float("inf")
            best_yellow = None
            for yx, yy, ya in yellows:
                d = np.sqrt((bx - yx) ** 2 + (by - yy) ** 2)
                if d < best_dist:
                    best_dist = d
                    best_yellow = (yx, yy, ya)
            if best_yellow is None:
                continue
            yx, yy, ya = best_yellow
            sqrt_area = np.sqrt(ya)
            if sqrt_area < 1:
                continue
            # Project the yellow→boundary vector onto long axis to get signed dist
            dx = bx - yx
            dy = by - yy
            proj_dist = abs(dx * long_dir[0] + dy * long_dir[1])
            ratio = proj_dist / sqrt_area
            if 0.3 < ratio < 10.0:  # sanity bounds
                ratios.append(ratio)

    # Compute median ratio
    median_ratio = None
    if ratios:
        ratios.sort()
        n = len(ratios)
        median_ratio = ratios[n // 2] if n % 2 == 1 else (ratios[n // 2 - 1] + ratios[n // 2]) / 2.0

    # Compute average polygon (only from 4-point polygons for consistency)
    avg_polygon = None
    four_pt_polys = [p for p in all_polys if len(p) == 4]
    if four_pt_polys:
        # Sort each polygon in angle order from centroid before averaging
        sorted_polys = []
        for poly in four_pt_polys:
            cx = sum(p[0] for p in poly) / 4
            cy = sum(p[1] for p in poly) / 4
            sp = sorted(poly, key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
            sorted_polys.append(sp)
        avg_polygon = []
        for i in range(4):
            ax = int(round(sum(p[i][0] for p in sorted_polys) / len(sorted_polys)))
            ay = int(round(sum(p[i][1] for p in sorted_polys) / len(sorted_polys)))
            avg_polygon.append([ax, ay])

    return median_ratio, avg_polygon


def _detect_cone_candidates(
    img: np.ndarray,
    offset_ratio: float | None = None,
    fallback_polygon: list[list[int]] | None = None,
) -> tuple[list[list], list[list[int]]]:
    """Detect game area boundary from cone markers on pool edge.

    Each side has: Red - Red - Yellow - [center] - Yellow - Red - Red.
    The boundary goes at the outer red, estimated by offsetting from the
    yellow along the pool's long axis (offset scales with apparent cone size).

    Args:
      offset_ratio: calibrated multiplier for sqrt(yellow_area) → boundary offset.
                    If None, uses default 2.0. Learned from user corrections.
      fallback_polygon: average polygon from saved masks, used when detection
                        finds fewer than 4 boundary points.

    Returns (all_markers, auto_polygon) where:
      all_markers: list of [x, y, color] for display (yellows + estimated reds)
      auto_polygon: list of [x, y] — game area boundary polygon
    """
    from wpv.tracking.detector import detect_pool_mask

    ratio = offset_ratio if offset_ratio is not None else 2.0

    pool_mask = detect_pool_mask(img)
    if pool_mask is None:
        if fallback_polygon:
            return [], fallback_polygon
        return [], []

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_img, w_img = img.shape[:2]

    # Pool geometry
    M = cv2.moments(pool_mask)
    if M["m00"] == 0:
        if fallback_polygon:
            return [], fallback_polygon
        return [], []
    pcx = int(M["m10"] / M["m00"])
    pcy = int(M["m01"] / M["m00"])

    cts, _ = cv2.findContours(
        pool_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    pool_ct = max(cts, key=cv2.contourArea)
    rect = cv2.minAreaRect(pool_ct)
    rw, rh = rect[1]
    angle = rect[2]
    long_angle = np.radians(angle + 90) if rw < rh else np.radians(angle)
    long_dir = np.array([np.cos(long_angle), np.sin(long_angle)])

    # Wide band around pool edge for yellow detection (80px out, 60px in)
    k_out = np.ones((80, 80), np.uint8)
    k_in = np.ones((60, 60), np.uint8)
    dilated = cv2.dilate(pool_mask, k_out)
    eroded = cv2.erode(pool_mask, k_in)
    band = cv2.subtract(dilated, eroded)

    # Find yellow cones
    yellow_mask = cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))
    yellow_near = cv2.bitwise_and(yellow_mask, band)
    yellow_near = cv2.morphologyEx(
        yellow_near, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    contours, _ = cv2.findContours(
        yellow_near, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    yellows_raw: list[tuple[int, int, float]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 15:
            continue
        m = cv2.moments(c)
        if m["m00"] > 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            yellows_raw.append((cx, cy, area))

    # Cluster nearby yellows (same cone, fragmented contours)
    yellows_raw.sort(key=lambda x: -x[2])
    yellows: list[tuple[int, int, float]] = []
    for yx, yy, ya in yellows_raw:
        if not any(
            np.sqrt((yx - ex) ** 2 + (yy - ey) ** 2) < 80
            for ex, ey, _ in yellows
        ):
            yellows.append((yx, yy, ya))

    if len(yellows) < 2:
        if fallback_polygon:
            return [], fallback_polygon
        return [], []

    # Project each yellow onto pool long axis, keep 2 per end (4 total)
    projections = []
    for yx, yy, ya in yellows:
        proj = (yx - pcx) * long_dir[0] + (yy - pcy) * long_dir[1]
        projections.append((proj, yx, yy, ya))
    projections.sort()

    if len(projections) >= 4:
        goal_yellows = projections[:2] + projections[-2:]
    else:
        goal_yellows = projections

    # Estimate outer red position for each yellow
    # Offset along pool long axis, away from center, scaled by cone size
    all_markers: list[list] = []
    boundary: list[list[int]] = []

    for proj, yx, yy, ya in goal_yellows:
        all_markers.append([yx, yy, "yellow"])
        sign = 1 if proj > 0 else -1
        offset = ratio * np.sqrt(ya)
        bx = int(yx + sign * long_dir[0] * offset)
        by = int(yy + sign * long_dir[1] * offset)
        bx = max(0, min(w_img - 1, bx))
        by = max(0, min(h_img - 1, by))
        boundary.append([bx, by])
        all_markers.append([bx, by, "red"])

    # Sort boundary into polygon order
    if len(boundary) >= 3:
        cx = sum(p[0] for p in boundary) / len(boundary)
        cy = sum(p[1] for p in boundary) / len(boundary)
        boundary.sort(key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))

    # If we got fewer than 4 points and have a fallback, fill in missing corners
    if len(boundary) < 4 and fallback_polygon and len(fallback_polygon) == 4:
        # Score each fallback corner by min distance to any detected point,
        # and add the most distant ones (most likely missing) up to 4 total
        fb_scored = []
        for fp in fallback_polygon:
            min_d = min(
                (np.sqrt((fp[0] - bp[0]) ** 2 + (fp[1] - bp[1]) ** 2)
                 for bp in boundary),
                default=float("inf"),
            )
            fb_scored.append((min_d, fp))
        fb_scored.sort(reverse=True)  # most distant first
        needed = 4 - len(boundary)
        for _, fp in fb_scored[:needed]:
            boundary.append(fp)
            all_markers.append([fp[0], fp[1], "red"])
        # Re-sort
        if len(boundary) >= 3:
            cx = sum(p[0] for p in boundary) / len(boundary)
            cy = sum(p[1] for p in boundary) / len(boundary)
            boundary.sort(key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))

    return all_markers, boundary


def _register_mask_routes(app, prep_data: dict) -> None:
    """Register game area mask drawing routes on a Flask app."""
    from flask import jsonify, request, send_file

    clips = prep_data.get("clips", [])
    # Build clip_name -> first frame mapping
    clip_first_frame: dict[str, str] = {}
    for f in prep_data["frames"]:
        cname = f"clip_{f['clip_idx']:03d}"
        if cname not in clip_first_frame:
            clip_first_frame[cname] = f["name"]

    def _clips_needing_masks() -> list[int]:
        """Return clip indices that don't have masks yet."""
        game_masks = _load_game_masks()
        missing = []
        for c in clips:
            cname = f"clip_{c['clip_idx']:03d}"
            if cname not in game_masks:
                missing.append(c["clip_idx"])
        return missing

    # Cache cone detections per clip (expensive, only compute once)
    cone_cache: dict[str, tuple[list, list]] = {}
    # Calibration state — recomputed when masks are saved
    calib_state: dict[str, object] = {"ratio": None, "avg_poly": None}

    def _refresh_calibration() -> None:
        """Recompute calibration from all saved masks."""
        ratio, avg_poly = _calibrate_from_saved_masks(prep_data)
        calib_state["ratio"] = ratio
        calib_state["avg_poly"] = avg_poly

    # Initial calibration from whatever masks exist
    _refresh_calibration()

    def _get_cone_data(clip_name: str) -> tuple[list, list]:
        if clip_name not in cone_cache:
            frame_name = clip_first_frame.get(clip_name)
            if frame_name:
                frame_path = FRAMES_DIR / f"{frame_name}.jpg"
                if frame_path.exists():
                    frame_img = cv2.imread(str(frame_path))
                    if frame_img is not None:
                        cone_cache[clip_name] = _detect_cone_candidates(
                            frame_img,
                            offset_ratio=calib_state["ratio"],
                            fallback_polygon=calib_state["avg_poly"],
                        )
                    else:
                        cone_cache[clip_name] = ([], [])
                else:
                    cone_cache[clip_name] = ([], [])
            else:
                cone_cache[clip_name] = ([], [])
        return cone_cache[clip_name]

    @app.route("/mask/<int:clip_idx>")
    def mask_page(clip_idx: int):
        if clip_idx < 0 or clip_idx >= len(clips):
            return "Invalid clip index", 404
        clip_name = f"clip_{clip_idx:03d}"
        game_masks = _load_game_masks()
        existing_poly = game_masks.get(clip_name)
        missing = _clips_needing_masks()
        done_count = len(clips) - len(missing)

        all_cones, auto_poly = _get_cone_data(clip_name)

        # Use existing polygon if saved, otherwise auto-detected polygon
        display_poly = existing_poly if existing_poly else auto_poly

        html = MASK_HTML_TEMPLATE
        html = html.replace("{{ clip_idx }}", str(clip_idx))
        html = html.replace("{{ clip_name }}", clip_name)
        html = html.replace("{{ clip_num }}", str(done_count + 1))
        html = html.replace("{{ total_clips }}", str(len(clips)))
        html = html.replace(
            "{{ progress_pct }}",
            str(round(100 * done_count / max(1, len(clips)))),
        )
        html = html.replace(
            "{{ existing_poly_json | safe }}",
            json.dumps(display_poly) if display_poly else "null",
        )
        html = html.replace(
            "{{ cone_candidates_json | safe }}",
            json.dumps(all_cones),
        )
        # Info text with calibration status
        ratio = calib_state["ratio"]
        n_saved = len(game_masks)
        if ratio is not None:
            info = (
                f"Calibrated from {n_saved} saved masks (ratio={ratio:.2f}). "
                f"{len(auto_poly)} auto-detected points. Press Enter to save & advance."
            )
        else:
            info = (
                f"No calibration yet (default ratio=2.0). "
                f"{len(auto_poly)} auto-detected points. Click to adjust, Enter to save."
            )
        html = html.replace("{{ info_text }}", info)
        # Review mode: mask already saved (user is editing, not creating for the first time)
        html = html.replace("{{ review_mode }}", "true" if existing_poly else "false")
        return html

    @app.route("/api/mask-frame/<int:clip_idx>")
    def mask_frame(clip_idx: int):
        clip_name = f"clip_{clip_idx:03d}"
        frame_name = clip_first_frame.get(clip_name)
        if not frame_name:
            return "No frame for clip", 404
        path = FRAMES_DIR / f"{frame_name}.jpg"
        if not path.exists():
            return "Frame not found", 404
        return send_file(path.resolve(), mimetype="image/jpeg")

    @app.route("/api/save-mask", methods=["POST"])
    def api_save_mask():
        data = request.get_json()
        clip_name = data.get("clip_name")
        polygon = data.get("polygon")
        if not clip_name or not polygon or len(polygon) < 3:
            return jsonify({"ok": False, "error": "Need clip_name and >= 3 points"})

        game_masks = _load_game_masks()
        game_masks[clip_name] = polygon
        GAME_MASKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        GAME_MASKS_PATH.write_text(json.dumps(game_masks, indent=2))

        # Recalibrate from all saved masks (including this new one)
        _refresh_calibration()
        # Clear cone cache for unsaved clips so they pick up the new calibration
        saved_clips = set(game_masks.keys())
        stale = [k for k in cone_cache if k not in saved_clips]
        for k in stale:
            del cone_cache[k]

        # Find next clip: if reviewing (came from /masks), advance sequentially;
        # otherwise go to next missing mask
        review_mode = data.get("review", False)
        clip_idx = next(
            (c["clip_idx"] for c in clips if f"clip_{c['clip_idx']:03d}" == clip_name),
            None,
        )
        if review_mode and clip_idx is not None and clip_idx + 1 < len(clips):
            next_clip = clip_idx + 1
        else:
            missing = _clips_needing_masks()
            next_clip = missing[0] if missing else None
        return jsonify({"ok": True, "next_clip": next_clip})

    @app.route("/api/masks-status")
    def api_masks_status():
        missing = _clips_needing_masks()
        return jsonify({
            "total_clips": len(clips),
            "missing": missing,
            "all_done": len(missing) == 0,
        })

    @app.route("/masks")
    def masks_overview():
        """Overview page to review/edit all game area masks."""
        game_masks = _load_game_masks()
        rows = ""
        for c in clips:
            ci = c["clip_idx"]
            cname = f"clip_{ci:03d}"
            has_mask = cname in game_masks
            status = "saved" if has_mask else "missing"
            color = "#2a9d8f" if has_mask else "#e94560"
            npts = len(game_masks[cname]) if has_mask else 0
            rows += (
                f'<tr><td><a href="/mask/{ci}" style="color:#0af">{cname}</a></td>'
                f'<td style="color:{color}">{status}</td>'
                f'<td>{npts} pts</td>'
                f'<td>{c.get("file","")}</td></tr>\n'
            )
        return f"""<!doctype html><html><head>
<meta charset="utf-8"><title>Game Area Masks</title>
<style>
body {{ background:#1a1a2e; color:#eee; font-family:system-ui; padding:20px; }}
h1 {{ margin:0 0 16px; }}
a {{ color:#0af; }}
table {{ border-collapse:collapse; width:100%; }}
th,td {{ text-align:left; padding:6px 12px; border-bottom:1px solid #333; }}
th {{ color:#888; font-size:13px; }}
.back {{ margin-bottom:16px; display:inline-block; }}
</style></head><body>
<a href="/" class="back">Back to Labeling</a>
<h1>Game Area Masks ({len(game_masks)}/{len(clips)} clips)</h1>
<p style="color:#888;font-size:13px;">Click a clip name to review/edit its mask.</p>
<table><tr><th>Clip</th><th>Status</th><th>Points</th><th>File</th></tr>
{rows}</table></body></html>"""


def cmd_batch(args: argparse.Namespace) -> None:
    """Run iterative batch labeling workflow as a single long-running server."""
    import threading

    from wpv.tracking.detector import HSV_BANDS, train_classifier, tune_hsv_from_labels

    try:
        from flask import Flask, jsonify, redirect, request
    except ImportError:
        print("ERROR: Flask not installed. Run: pip install flask", file=sys.stderr)
        sys.exit(1)

    if not PREP_PATH.exists():
        print(f"ERROR: {PREP_PATH} not found. Run prep first.", file=sys.stderr)
        sys.exit(1)

    schedule = [int(x) for x in args.schedule.split(",")]

    # Load or create batch state
    if BATCH_STATE_PATH.exists():
        batch_state = json.loads(BATCH_STATE_PATH.read_text())
        if batch_state["schedule"] != schedule:
            print("WARNING: Schedule changed. Using saved state's schedule.")
            schedule = batch_state["schedule"]
    else:
        batch_state = {
            "schedule": schedule,
            "current_batch": 0,
            "batches": [],
        }

    # Shared mutable state for the server
    state = {
        "annotations": _load_annotations(),
        "prep_data": json.loads(PREP_PATH.read_text()),
        "batch_state": batch_state,
        "schedule": schedule,
        "already_selected": set(),
        "current_frames": [],       # frames for current batch
        "phase": "idle",            # "labeling", "training", "done"
        "train_progress": "",       # status text shown to UI during training
    }

    # Collect already-selected frames from previous batches
    for b in batch_state["batches"]:
        state["already_selected"].update(b["frame_names"])

    # Undo stack
    undo_stack: list[tuple[str, object]] = []
    _SENTINEL = object()
    MAX_UNDO = 20

    def save_annotations() -> None:
        ANNOTATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        ANNOTATIONS_PATH.write_text(json.dumps({
            "version": 2,
            "frames": state["annotations"],
        }, indent=2))

    def _save_batch_state() -> None:
        BATCH_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BATCH_STATE_PATH.write_text(json.dumps(state["batch_state"], indent=2))

    def _start_batch(batch_idx: int) -> dict:
        """Select frames and prepare a batch. Returns batch info dict."""
        bs = state["batch_state"]
        if batch_idx >= len(state["schedule"]):
            state["phase"] = "done"
            return {"done": True}

        batch_size = state["schedule"][batch_idx]
        frame_names = _select_batch_frames(
            batch_idx, batch_size, state["prep_data"],
            state["annotations"], state["already_selected"],
        )
        if not frame_names:
            state["phase"] = "done"
            return {"done": True}

        state["already_selected"].update(frame_names)

        # Re-run HSV with tuned params for batch > 0
        if batch_idx > 0:
            state["prep_data"] = _rerun_hsv_for_frames(
                frame_names, state["prep_data"]
            )

        # Record in batch state
        batch_record = {
            "batch_id": batch_idx,
            "frame_names": frame_names,
            "status": "labeling",
        }
        if batch_idx < len(bs["batches"]):
            bs["batches"][batch_idx] = batch_record
        else:
            bs["batches"].append(batch_record)
        bs["current_batch"] = batch_idx
        _save_batch_state()

        # Update current frames for serving
        prep_lookup = {f["name"]: f for f in state["prep_data"]["frames"]}
        state["current_frames"] = [
            prep_lookup[n] for n in frame_names if n in prep_lookup
        ]
        state["phase"] = "labeling"
        undo_stack.clear()

        print(f"\nBATCH {batch_idx} — {len(frame_names)} frames")
        return {
            "done": False,
            "batch_id": batch_idx,
            "num_frames": len(frame_names),
        }

    def _run_training() -> None:
        """Run HSV tuning + CNN training in background thread."""
        annotations = state["annotations"]
        prep_data = state["prep_data"]

        # HSV tuning
        state["train_progress"] = "Tuning HSV bands..."
        if HSV_PARAMS_PATH.exists():
            hsv_data = json.loads(HSV_PARAMS_PATH.read_text())
            current_bands = [
                (tuple(b["lo"]), tuple(b["hi"])) for b in hsv_data["bands"]
            ]
        else:
            current_bands = list(HSV_BANDS)

        new_bands = tune_hsv_from_labels(
            FRAMES_DIR, annotations, prep_data, current_bands
        )
        HSV_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
        HSV_PARAMS_PATH.write_text(json.dumps({
            "bands": [{"lo": list(lo), "hi": list(hi)} for lo, hi in new_bands],
        }, indent=2))
        band_strs = [
            f"H:{lo[0]}-{hi[0]} S:{lo[1]}-{hi[1]} V:{lo[2]}-{hi[2]}"
            for lo, hi in new_bands
        ]
        state["train_progress"] = (
            f"HSV: {len(new_bands)} bands saved. Training CNN..."
        )
        print(f"  HSV: {len(new_bands)} bands — {', '.join(band_strs)}")

        # CNN training
        model_path = MODEL_DIR / "ball_classifier.pth"
        pretrained = model_path if model_path.exists() else None
        result = train_classifier(
            FRAMES_DIR, annotations, prep_data, model_path,
            pretrained_path=pretrained,
        )
        if result.get("skipped"):
            state["train_progress"] = (
                f"HSV tuned. CNN skipped: {result['reason']}"
            )
            print(f"  CNN skipped: {result['reason']}")
        else:
            state["train_progress"] = (
                f"Done! {result['n_pos']} pos, {result['n_neg']} neg, "
                f"loss={result['best_loss']:.4f}, acc={result['final_accuracy']:.1%}"
            )
            print(
                f"  CNN: {result['n_pos']} pos, {result['n_neg']} neg, "
                f"loss={result['best_loss']:.4f}, acc={result['final_accuracy']:.1%}"
            )

        # Mark batch complete, advance
        bs = state["batch_state"]
        batch_idx = bs["current_batch"]
        bs["batches"][batch_idx]["status"] = "complete"
        bs["current_batch"] = batch_idx + 1
        _save_batch_state()

        # Start next batch
        _start_batch(batch_idx + 1)

    # Check for missing game area masks before starting
    clips = state["prep_data"].get("clips", [])
    game_masks = _load_game_masks()
    missing_masks = [
        c["clip_idx"] for c in clips
        if f"clip_{c['clip_idx']:03d}" not in game_masks
    ]
    needs_masks = len(missing_masks) > 0
    if needs_masks:
        state["phase"] = "masking"
        print(f"  {len(missing_masks)} clips need game area masks — showing mask UI first")
    else:
        # Start first batch
        start_batch = batch_state["current_batch"]
        _start_batch(start_batch)
        if state["phase"] == "done":
            print("No frames to label.")
            return

    # Build Flask app
    app = Flask(__name__)

    # Register mask drawing routes
    _register_mask_routes(app, state["prep_data"])

    @app.route("/")
    def index():
        # If masks are still needed, redirect to mask UI
        gm = _load_game_masks()
        clips_list = state["prep_data"].get("clips", [])
        still_missing = [
            c["clip_idx"] for c in clips_list
            if f"clip_{c['clip_idx']:03d}" not in gm
        ]
        if still_missing:
            return redirect(f"/mask/{still_missing[0]}")
        # If we haven't started a batch yet (just finished masking), start now
        if state["phase"] == "masking":
            start_b = state["batch_state"]["current_batch"]
            _start_batch(start_b)
            if state["phase"] == "done":
                return "No frames to label.", 200
        return redirect("/label/0")

    @app.route("/label/<int:idx>")
    def label_page(idx: int):
        frames = state["current_frames"]
        if not frames:
            return "No batch active", 404
        idx = max(0, min(idx, len(frames) - 1))
        frame_data = [
            {"name": f["name"], "candidates": f["candidates"]}
            for f in frames
        ]
        html = HTML_TEMPLATE
        html = html.replace("{{ frame_data_json | safe }}", json.dumps(frame_data))
        html = html.replace("{{ total }}", str(len(frames)))
        html = html.replace("{{ idx }}", str(idx))
        html = html.replace("{{ idx+1 }}", str(idx + 1))
        html = html.replace("{{ frame_name }}", frames[idx]["name"])
        html = html.replace(
            "{{ annotations_json | safe }}", json.dumps(state["annotations"])
        )
        return html

    @app.route("/api/frame/<name>")
    def serve_frame(name: str):
        from flask import send_file
        safe = Path(name).name
        path = FRAMES_DIR / safe
        if not path.exists():
            return "Not found", 404
        return send_file(path.resolve(), mimetype="image/jpeg")

    @app.route("/api/annotate", methods=["POST"])
    def api_annotate():
        data = request.get_json()
        frame_name = data["frame"]
        value = data["value"]
        prev = state["annotations"].get(frame_name, _SENTINEL)
        undo_stack.append((frame_name, prev))
        if len(undo_stack) > MAX_UNDO:
            undo_stack.pop(0)
        state["annotations"][frame_name] = value
        save_annotations()
        return jsonify({"ok": True, "annotations": state["annotations"]})

    @app.route("/api/undo", methods=["POST"])
    def api_undo():
        if not undo_stack:
            return jsonify({"undone": None, "annotations": state["annotations"]})
        frame_name, prev = undo_stack.pop()
        if prev is _SENTINEL:
            state["annotations"].pop(frame_name, None)
        else:
            state["annotations"][frame_name] = prev
        save_annotations()
        return jsonify({"undone": frame_name, "annotations": state["annotations"]})

    @app.route("/api/progress")
    def api_progress():
        frames = state["current_frames"]
        return jsonify({
            "total": len(frames),
            "labeled": sum(1 for f in frames if f["name"] in state["annotations"]),
            "remaining": sum(
                1 for f in frames if f["name"] not in state["annotations"]
            ),
        })

    @app.route("/api/batch-status")
    def api_batch_status():
        bs = state["batch_state"]
        return jsonify({
            "phase": state["phase"],
            "batch_id": bs["current_batch"],
            "schedule": state["schedule"],
            "total_batches": len(state["schedule"]),
            "total_annotations": len(state["annotations"]),
            "train_progress": state["train_progress"],
            "num_frames": len(state["current_frames"]),
        })

    @app.route("/api/finish-batch", methods=["POST"])
    def api_finish_batch():
        if state["phase"] != "labeling":
            return jsonify({"ok": False, "error": "not in labeling phase"})
        save_annotations()
        state["phase"] = "training"
        state["train_progress"] = "Starting training..."
        # Run training in background thread
        t = threading.Thread(target=_run_training, daemon=True)
        t.start()
        return jsonify({"ok": True})

    @app.route("/api/shutdown", methods=["POST"])
    def api_shutdown():
        save_annotations()
        import signal
        threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
        return jsonify({"ok": True})

    print(f"Batch labeling UI: http://0.0.0.0:{args.port}/")
    print(f"  Schedule: {schedule}")
    if needs_masks:
        print(f"  Drawing game area masks for {len(missing_masks)} clips first")
    else:
        print(f"  Starting at batch {batch_state['current_batch']}")
    app.run(host="0.0.0.0", port=args.port, debug=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Ball labeling tool")
    sub = parser.add_subparsers(dest="command")

    p_prep = sub.add_parser("prep", help="Extract frames and run HSV detection")
    p_prep.add_argument("--count", type=int, default=500, help="Target number of frames")
    p_prep.add_argument("--force", action="store_true", help="Overwrite existing annotations")

    p_serve = sub.add_parser("serve", help="Launch labeling web UI")
    p_serve.add_argument("--port", type=int, default=5001, help="HTTP port")

    sub.add_parser("train", help="Tune HSV params and train CNN from annotations")

    p_batch = sub.add_parser("batch", help="Iterative batch labeling workflow")
    p_batch.add_argument(
        "--schedule", type=str, default="50,50,50,50,50,50",
        help="Comma-separated batch sizes (default: 50,50,50,50,50,50)"
    )
    p_batch.add_argument("--port", type=int, default=5001, help="HTTP port")

    args = parser.parse_args()
    if args.command == "prep":
        cmd_prep(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "batch":
        cmd_batch(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
