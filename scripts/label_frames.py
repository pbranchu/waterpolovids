#!/usr/bin/env python3
"""Extract frames from LRV files for labeling.

Standalone script (not part of wpv CLI) to sample frames from video clips
and prepare them for annotation.

Usage:
    python scripts/label_frames.py VIDEO_PATH --count 100 --output data/labeling/frames/
    python scripts/label_frames.py --clips video1.lrv video2.lrv --count 50 --output frames/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    count: int = 100,
    strategy: str = "uniform",
    clip_prefix: str = "clip_000",
) -> list[Path]:
    """Extract frames from a video file.

    Parameters
    ----------
    video_path   : path to input video (LRV or any OpenCV-readable format)
    output_dir   : directory to save extracted PNGs
    count        : number of frames to extract
    strategy     : 'uniform', 'random', or 'diverse'
    clip_prefix  : prefix for output filenames

    Returns
    -------
    List of paths to saved frame images.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise RuntimeError(f"Cannot determine frame count for: {video_path}")

    count = min(count, total_frames)
    frame_indices = _select_indices(cap, total_frames, count, strategy)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for idx in sorted(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        fname = out / f"{clip_prefix}_frame_{idx:06d}.png"
        cv2.imwrite(str(fname), frame)
        saved.append(fname)

    cap.release()
    return saved


def _select_indices(
    cap: cv2.VideoCapture,
    total: int,
    count: int,
    strategy: str,
) -> list[int]:
    """Choose which frame indices to extract."""
    if strategy == "uniform":
        step = max(1, total // count)
        return [i * step for i in range(count) if i * step < total]

    if strategy == "random":
        return sorted(random.sample(range(total), count))

    if strategy == "diverse":
        return _diverse_indices(cap, total, count)

    raise ValueError(f"Unknown strategy: {strategy!r} (use 'uniform', 'random', or 'diverse')")


def _diverse_indices(cap: cv2.VideoCapture, total: int, count: int) -> list[int]:
    """Select frames with highest scene-change scores (frame diff)."""
    # Sample diffs at regular intervals to find scene changes
    sample_step = max(1, total // (count * 10))
    diffs: list[tuple[float, int]] = []

    prev_gray = None
    for idx in range(0, total, sample_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = float(np.mean(cv2.absdiff(prev_gray, gray)))
            diffs.append((diff, idx))
        prev_gray = gray

    # Pick top-N most different frames
    diffs.sort(reverse=True)
    selected = [idx for _, idx in diffs[:count]]

    # If not enough diverse frames, pad with uniform
    if len(selected) < count:
        step = max(1, total // (count - len(selected) + 1))
        existing = set(selected)
        for i in range(0, total, step):
            if i not in existing:
                selected.append(i)
            if len(selected) >= count:
                break

    return sorted(selected[:count])


def generate_annotations_stub(frame_paths: list[Path], output_path: Path) -> None:
    """Write a stub annotations.json with empty labels for each frame."""
    annotations = {
        "frames": [
            {"file": str(p.name), "labels": []}
            for p in frame_paths
        ]
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(annotations, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract frames from LRV files for labeling")
    parser.add_argument("video", nargs="?", help="Path to a single video file")
    parser.add_argument("--clips", nargs="+", help="Paths to multiple video clips")
    parser.add_argument("--count", type=int, default=100, help="Number of frames per clip")
    parser.add_argument("--strategy", default="uniform", choices=["uniform", "random", "diverse"])
    parser.add_argument("--output", "-o", default="data/labeling/frames", help="Output directory")

    args = parser.parse_args(argv)

    videos: list[Path] = []
    if args.clips:
        videos = [Path(c) for c in args.clips]
    elif args.video:
        videos = [Path(args.video)]
    else:
        parser.error("Provide a video path or use --clips")

    all_frames: list[Path] = []
    for i, vpath in enumerate(videos):
        prefix = f"clip_{i:03d}"
        print(f"Extracting {args.count} frames from {vpath} (strategy={args.strategy})")
        frames = extract_frames(vpath, args.output, args.count, args.strategy, prefix)
        all_frames.extend(frames)
        print(f"  Saved {len(frames)} frames")

    # Generate stub annotations
    ann_path = Path(args.output) / "annotations.json"
    generate_annotations_stub(all_frames, ann_path)
    print(f"Annotations stub: {ann_path}")
    print(f"Total frames extracted: {len(all_frames)}")


if __name__ == "__main__":
    main()
