"""End-to-end pipeline orchestrator: match directory → processed output."""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from wpv.config import settings
from wpv.db import MatchRecord, get_record, update_field, update_status, upsert_record


class Stage(enum.Enum):
    DETECT = "detect"
    DECODE = "decode"
    TRACK = "track"
    RENDER = "render"
    HIGHLIGHTS = "highlights"
    QUALITY_CHECK = "quality-check"
    UPLOAD = "upload"


ALL_STAGES = list(Stage)


@dataclass
class StageResult:
    stage: Stage
    success: bool
    skipped: bool = False
    output_path: Path | None = None
    error: str | None = None


@dataclass
class PipelineResult:
    match_id: str
    source_dir: Path
    stages: list[StageResult] = field(default_factory=list)
    success: bool = False
    youtube_url: str | None = None
    youtube_urls: list[str] = field(default_factory=list)


def run_pipeline(
    match_dir: Path,
    stages: list[Stage] | None = None,
    skip_upload: bool = False,
    db_path: Path | None = None,
    progress_callback: Callable[[Stage, str], None] | None = None,
) -> PipelineResult:
    """Run the full pipeline on a match directory.

    Each stage is idempotent — it checks for existing output before running.
    Pipeline stops on first failed stage.
    """
    match_dir = Path(match_dir).resolve()
    match_id = match_dir.name
    work_dir = match_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    active_stages = stages or ALL_STAGES
    if skip_upload:
        active_stages = [s for s in active_stages if s != Stage.UPLOAD]

    result = PipelineResult(match_id=match_id, source_dir=match_dir)

    # Register in DB
    record = get_record(match_id, db_path)
    if record is None:
        record = MatchRecord(
            match_id=match_id,
            source_path=str(match_dir),
            status="processing",
            started_at=_now(),
            work_dir=str(work_dir),
        )
        upsert_record(record, db_path)
    else:
        update_status(match_id, "processing", db_path=db_path)
        update_field(match_id, "started_at", _now(), db_path)

    stage_funcs = {
        Stage.DETECT: _stage_detect,
        Stage.DECODE: _stage_decode,
        Stage.TRACK: _stage_track,
        Stage.RENDER: _stage_render,
        Stage.HIGHLIGHTS: _stage_highlights,
        Stage.QUALITY_CHECK: _stage_quality_check,
        Stage.UPLOAD: _stage_upload,
    }

    for stage in active_stages:
        if progress_callback:
            progress_callback(stage, "starting")

        try:
            if stage == Stage.TRACK and progress_callback:
                sr = stage_funcs[stage](match_dir, work_dir, match_id, db_path, progress_callback)
            else:
                sr = stage_funcs[stage](match_dir, work_dir, match_id, db_path)
        except Exception as e:
            sr = StageResult(stage=stage, success=False, error=str(e))

        result.stages.append(sr)

        if progress_callback:
            status_msg = "skipped" if sr.skipped else ("done" if sr.success else f"failed: {sr.error}")
            progress_callback(stage, status_msg)

        if not sr.success and not sr.skipped:
            update_status(match_id, "failed", error_message=sr.error, db_path=db_path)
            result.success = False
            return result

    # All stages passed
    update_status(match_id, "completed", db_path=db_path)
    update_field(match_id, "completed_at", _now(), db_path)
    result.success = True

    # Grab youtube URLs if uploaded
    rec = get_record(match_id, db_path)
    if rec and rec.youtube_url:
        result.youtube_url = rec.youtube_url
        # youtube_url is stored as JSON array of URLs
        try:
            result.youtube_urls = json.loads(rec.youtube_url)
        except (json.JSONDecodeError, TypeError):
            result.youtube_urls = [rec.youtube_url]

    return result


# ---------------------------------------------------------------------------
# Per-stage functions (idempotent)
# ---------------------------------------------------------------------------


def _stage_detect(match_dir: Path, work_dir: Path, match_id: str, db_path) -> StageResult:
    """Detect video format for all clips. Always runs (stateless)."""
    from wpv.ingest.detect_format import detect_format

    videos = _find_videos(match_dir)
    if not videos:
        return StageResult(stage=Stage.DETECT, success=False, error="No .mp4 files found")

    for v in videos:
        detect_format(v)

    return StageResult(stage=Stage.DETECT, success=True)


def _stage_decode(match_dir: Path, work_dir: Path, match_id: str, db_path) -> StageResult:
    """Decode/symlink videos into work dir. Skips if work/*.mp4 exist."""
    from wpv.ingest.detect_format import detect_format
    from wpv.ingest.stitch import prepare_equirect

    existing = [p for p in work_dir.glob("*.mp4") if "_render" not in p.stem and p.stem != "output"]
    if existing:
        return StageResult(stage=Stage.DECODE, success=True, skipped=True,
                           output_path=existing[0])

    videos = _find_videos(match_dir)
    if not videos:
        return StageResult(stage=Stage.DECODE, success=False, error="No .mp4 files found")

    last_out = None
    for v in videos:
        info = detect_format(v)
        last_out = prepare_equirect(info, work_dir)

    return StageResult(stage=Stage.DECODE, success=True, output_path=last_out)


def _stage_track(match_dir: Path, work_dir: Path, match_id: str, db_path, progress_callback=None) -> StageResult:
    """Track ball in each work/*.mp4 clip, in parallel.

    Each clip gets its own tracker thread.  A shared BallDetector (thread-safe
    for inference) avoids redundant model loading.  Clips whose .json already
    exists are skipped.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from wpv.tracking.detector import BallDetector
    from wpv.tracking.track_io import load_track, save_track
    from wpv.tracking.tracker import BallTracker

    work_videos = sorted(p for p in work_dir.glob("*.mp4") if "_render" not in p.stem and p.stem != "output")
    if not work_videos:
        return StageResult(stage=Stage.TRACK, success=False, error="No decoded videos in work/")

    # Fail fast if model or reference image is missing
    missing = []
    if not settings.ball_model_path.exists():
        missing.append(f"CNN model not found: {settings.ball_model_path}")
    if not settings.ball_ref_image_path.exists():
        missing.append(f"Reference image not found: {settings.ball_ref_image_path}")
    if missing:
        return StageResult(stage=Stage.TRACK, success=False, error="; ".join(missing))

    # Shared detector — the CNN model and histogram scorer are read-only
    # during inference, so safe to share across threads.
    detector = BallDetector(
        model_path=settings.ball_model_path,
        ref_image_path=settings.ball_ref_image_path,
        min_area=settings.min_ball_px,
        max_area=settings.max_ball_px,
        confidence_threshold=settings.detection_confidence_threshold,
    )

    num_clips = len(work_videos)
    # Per-clip progress state, guarded by a lock.
    # Skipped clips (JSON exists) are pre-filled as 100%.
    clip_state: dict[int, tuple[int, int, str]] = {}
    skipped: set[int] = set()
    state_lock = threading.Lock()

    # Pre-scan for already-completed clips
    for ci, vp in enumerate(work_videos):
        if (work_dir / f"{vp.stem}.json").exists():
            skipped.add(ci)
            clip_state[ci] = (1, 1, "done")

    def _report_aggregate():
        """Build an aggregate progress message from all clips."""
        if not progress_callback:
            return
        with state_lock:
            if not clip_state:
                return
            total_pct = 0.0
            parts = []
            for ci in range(num_clips):
                if ci in skipped:
                    total_pct += 100.0
                    continue
                if ci not in clip_state:
                    continue
                fi, ft, sn = clip_state[ci]
                pct = fi / ft * 100 if ft > 0 else 0
                total_pct += pct
                parts.append(f"c{ci+1}:{sn[:4]}")
            overall = total_pct / num_clips
        active = " ".join(parts) if parts else "waiting"
        progress_callback(Stage.TRACK, f"{num_clips} clips ({overall:.0f}%) [{active}]")

    def _track_one(clip_idx: int, vp: Path):
        track_json = work_dir / f"{vp.stem}.json"
        if track_json.exists():
            return clip_idx, load_track(track_json), track_json

        # Each thread gets its own BallTracker (has mutable per-track state)
        tracker = BallTracker(
            detector=detector,
            detection_scale=settings.track_detection_scale,
            loss_frames=settings.track_loss_frames,
            search_step_s=settings.search_forward_step_s,
            search_max_gap_s=settings.search_max_gap_s,
            rewind_step_s=settings.rewind_coarse_step_s,
            reacquire_persistence=settings.track_reacquire_persistence,
            gate_distance=settings.track_gate_distance,
            confidence_threshold=settings.detection_confidence_threshold,
        )

        def _clip_progress(frame_idx, frame_total, state_name):
            with state_lock:
                clip_state[clip_idx] = (frame_idx, frame_total, state_name)
            _report_aggregate()

        result = tracker.track_clip(
            vp, clip_name=vp.stem,
            progress_callback=_clip_progress if progress_callback else None,
        )
        save_track(result, track_json)
        return clip_idx, result, track_json

    max_workers = min(num_clips, settings.track_parallel_clips)
    results: dict[int, tuple] = {}
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_track_one, ci, vp): ci
            for ci, vp in enumerate(work_videos)
        }
        for future in as_completed(futures):
            ci = futures[future]
            try:
                clip_idx, result, track_json = future.result()
                results[clip_idx] = (result, track_json)
            except Exception as exc:
                errors.append(f"Clip {ci} ({work_videos[ci].name}): {exc}")

    if errors:
        return StageResult(stage=Stage.TRACK, success=False, error="; ".join(errors))

    # Aggregate results
    total_frames = 0
    tracking_frames = 0
    last_track_path = None
    for ci in sorted(results):
        result, track_json = results[ci]
        total_frames += result.frame_count
        tracking_frames += sum(1 for pt in result.points if pt.state == "tracking")
        last_track_path = track_json

    if last_track_path:
        update_field(match_id, "track_path", str(last_track_path), db_path)

    # Fail early if ball was never detected — don't waste time rendering
    if total_frames > 0:
        coverage = tracking_frames / total_frames * 100
        if coverage < settings.quality_min_track_coverage_pct:
            return StageResult(
                stage=Stage.TRACK, success=False,
                error=f"Ball tracking coverage too low: {coverage:.1f}% "
                      f"(need {settings.quality_min_track_coverage_pct}%)")

    return StageResult(stage=Stage.TRACK, success=True, output_path=last_track_path)


def _stage_render(match_dir: Path, work_dir: Path, match_id: str, db_path) -> StageResult:
    """Render crop-and-pan output for all tracked clips.

    Skips if work/output.mp4 exists.  For each track JSON in work/, renders
    a ``<stem>_render.mp4``.  If multiple renders exist, concatenates them
    into ``output.mp4``; if only one, renames it.

    Pool bounds are loaded from ``game_masks.json`` (in match_dir or the
    default config path) when available.
    """
    import subprocess

    from wpv.render.reframe import CropRenderer, load_game_mask_bounds
    from wpv.tracking.track_io import load_track

    output_mp4 = work_dir / "output.mp4"
    if output_mp4.exists():
        update_field(match_id, "render_path", str(output_mp4), db_path)
        return StageResult(stage=Stage.RENDER, success=True, skipped=True,
                           output_path=output_mp4)

    # Find all track JSONs (exclude segments.json and game_masks.json)
    skip_names = {"segments.json", "game_masks.json"}
    track_jsons = sorted(
        t for t in work_dir.glob("*.json") if t.name not in skip_names
    )
    if not track_jsons:
        return StageResult(stage=Stage.RENDER, success=False, error="No track JSONs found")

    # Locate game_masks.json — check match_dir first, then config default
    game_masks_path = match_dir / "game_masks.json"
    if not game_masks_path.exists():
        game_masks_path = Path(settings.game_masks_path)
    has_game_masks = game_masks_path.exists()

    # Render each tracked clip
    render_parts: list[Path] = []
    for track_path in track_jsons:
        render_mp4 = work_dir / f"{track_path.stem}_render.mp4"
        if render_mp4.exists():
            render_parts.append(render_mp4)
            continue

        video_path = work_dir / f"{track_path.stem}.mp4"
        if not video_path.exists():
            return StageResult(stage=Stage.RENDER, success=False,
                               error=f"Source video not found: {video_path}")

        track = load_track(track_path)

        # Derive clip_name from video filename: PRO_VID_*_NNN → clip_{NNN-1:03d}
        pool_bounds = None
        if has_game_masks:
            clip_name = _video_stem_to_clip_name(track_path.stem)
            if clip_name:
                try:
                    pool_bounds = load_game_mask_bounds(game_masks_path, clip_name)
                except KeyError:
                    pass  # fall back to auto-detection

        renderer = CropRenderer(
            video_path=video_path,
            track=track,
            output_path=render_mp4,
            crop_w=settings.crop_output_width,
            crop_h=settings.crop_output_height,
            alpha=settings.crop_smoothing_alpha,
            dead_zone=settings.crop_dead_zone_px,
            max_vel=settings.crop_max_velocity_px,
            codec=settings.crop_output_codec,
            crf=settings.crop_output_crf,
            preset=settings.crop_output_preset,
            pool_bounds=pool_bounds,
        )
        renderer.run()
        render_parts.append(render_mp4)

    # Concatenate all rendered parts into output.mp4
    # Keep individual renders for per-clip upload
    if len(render_parts) == 1:
        import shutil as _shutil
        _shutil.copy2(render_parts[0], output_mp4)
    else:
        concat_list = work_dir / "concat.txt"
        concat_list.write_text(
            "\n".join(f"file '{p.name}'" for p in render_parts)
        )
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            "-movflags", "+faststart",
            str(output_mp4),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return StageResult(stage=Stage.RENDER, success=False,
                               error=f"ffmpeg concat failed: {proc.stderr[-500:]}")

    update_field(match_id, "render_path", str(output_mp4), db_path)
    return StageResult(stage=Stage.RENDER, success=True, output_path=output_mp4)


def _stage_highlights(match_dir: Path, work_dir: Path, match_id: str, db_path) -> StageResult:
    """Extract highlights from ALL clips, build a single highlight reel.

    Skips if work/highlights.mp4 exists.  Scores every track JSON,
    offsets timestamps to the concatenated output.mp4 timeline, selects
    the best segments across the full game, and builds one montage.
    """
    import subprocess as _sp

    from wpv.render.highlights import (
        HighlightSegment,
        build_montage,
        save_segments,
        score_track,
        select_segments,
    )
    from wpv.tracking.track_io import load_track

    highlights_mp4 = work_dir / "highlights.mp4"
    if highlights_mp4.exists():
        update_field(match_id, "highlights_path", str(highlights_mp4), db_path)
        return StageResult(stage=Stage.HIGHLIGHTS, success=True, skipped=True,
                           output_path=highlights_mp4)

    output_mp4 = work_dir / "output.mp4"
    if not output_mp4.exists():
        return StageResult(stage=Stage.HIGHLIGHTS, success=False,
                           error="Rendered output.mp4 not found")

    # Find track JSONs (those with a matching source .mp4 in work/)
    track_jsons = sorted(
        t for t in work_dir.glob("*.json")
        if (work_dir / f"{t.stem}.mp4").exists()
    )
    if not track_jsons:
        return StageResult(stage=Stage.HIGHLIGHTS, success=False, error="No track JSONs found")

    # Score every clip and offset segments to the concatenated timeline
    all_segments: list[HighlightSegment] = []
    time_offset = 0.0
    last_fps = 25.0

    for track_path in track_jsons:
        track = load_track(track_path)
        last_fps = track.fps
        clip_duration = track.frame_count / track.fps if track.fps > 0 else 0.0

        scores = score_track(
            track,
            speed_sigma_threshold=settings.highlight_speed_sigma,
            direction_window_s=settings.highlight_direction_window_s,
            gap_reappear_bonus=settings.highlight_gap_reappear_bonus,
        )
        segments = select_segments(
            scores,
            fps=track.fps,
            threshold=settings.highlight_score_threshold,
            context_s=settings.highlight_context_s,
            min_segment_s=settings.highlight_min_duration_s,
            target_duration_s=settings.highlight_target_duration_s,
            max_segments=settings.highlight_max_segments,
        )

        # Offset segments to concatenated-video timeline
        for seg in segments:
            all_segments.append(HighlightSegment(
                start_frame=seg.start_frame + int(time_offset * track.fps),
                end_frame=seg.end_frame + int(time_offset * track.fps),
                start_s=seg.start_s + time_offset,
                end_s=seg.end_s + time_offset,
                peak_score=seg.peak_score,
                mean_score=seg.mean_score,
            ))

        # Use actual render duration for offset (more accurate than track frames)
        render_mp4 = work_dir / f"{track_path.stem}_render.mp4"
        if render_mp4.exists():
            try:
                probe = _sp.run(
                    ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                     "-of", "csv=p=0", str(render_mp4)],
                    capture_output=True, text=True, check=True,
                )
                time_offset += float(probe.stdout.strip())
            except Exception:
                time_offset += clip_duration
        else:
            time_offset += clip_duration

    # Re-rank all segments across clips by peak score, trim to target
    all_segments.sort(key=lambda s: s.peak_score, reverse=True)
    selected: list[HighlightSegment] = []
    total_dur = 0.0
    for seg in all_segments:
        if len(selected) >= settings.highlight_max_segments:
            break
        dur = seg.end_s - seg.start_s
        if total_dur + dur > settings.highlight_target_duration_s and selected:
            continue
        selected.append(seg)
        total_dur += dur
    selected.sort(key=lambda s: s.start_s)

    segments_json = work_dir / "segments.json"
    save_segments(selected, segments_json)
    update_field(match_id, "segments_path", str(segments_json), db_path)

    if selected:
        build_montage(
            selected,
            source_video=output_mp4,
            output_path=highlights_mp4,
            fps=last_fps,
            crossfade_s=settings.highlight_crossfade_s,
            codec=settings.crop_output_codec,
            crf=settings.crop_output_crf,
            preset=settings.crop_output_preset,
        )
        update_field(match_id, "highlights_path", str(highlights_mp4), db_path)
        return StageResult(stage=Stage.HIGHLIGHTS, success=True, output_path=highlights_mp4)

    return StageResult(stage=Stage.HIGHLIGHTS, success=True, output_path=segments_json)


def _stage_quality_check(match_dir: Path, work_dir: Path, match_id: str, db_path) -> StageResult:
    """Run quality gates. Always runs (cheap)."""
    from wpv.quality import run_quality_gates
    from wpv.render.highlights import load_segments
    from wpv.tracking.track_io import load_track

    track_jsons = sorted(
        t for t in work_dir.glob("*.json")
        if (work_dir / f"{t.stem}.mp4").exists()
    )
    if not track_jsons:
        return StageResult(stage=Stage.QUALITY_CHECK, success=False,
                           error="No track JSONs found")

    track = load_track(track_jsons[0])

    segments = None
    segments_json = work_dir / "segments.json"
    if segments_json.exists():
        segments = load_segments(segments_json)

    report = run_quality_gates(
        track,
        segments=segments,
        min_track_coverage_pct=settings.quality_min_track_coverage_pct,
        min_highlight_duration_s=settings.quality_min_highlight_duration_s,
    )

    update_field(match_id, "track_coverage_pct", report.track_coverage_pct, db_path)
    update_field(match_id, "quality_passed", report.passed, db_path)

    if not report.passed:
        reasons = []
        if not report.track_coverage_ok:
            reasons.append(f"coverage {report.track_coverage_pct}% < {settings.quality_min_track_coverage_pct}%")
        if not report.highlight_duration_ok:
            reasons.append(f"highlights {report.highlight_duration_s}s < {settings.quality_min_highlight_duration_s}s")
        return StageResult(stage=Stage.QUALITY_CHECK, success=False,
                           error=f"Quality gates failed: {'; '.join(reasons)}")

    return StageResult(stage=Stage.QUALITY_CHECK, success=True)


def _stage_upload(match_dir: Path, work_dir: Path, match_id: str, db_path) -> StageResult:
    """Upload per-clip renders + highlight reel to YouTube.

    For a game with N clips, uploads:
    - N individual clip renders (``{stem}_render.mp4``)
    - 1 highlight reel (``highlights.mp4``) named "{title} - Highlights"

    Single-clip title = game name.  Multi-clip titles = "{name} - Part N".
    Skips if DB already has youtube_video_id.
    """
    rec = get_record(match_id, db_path)
    if rec and rec.youtube_video_id:
        return StageResult(stage=Stage.UPLOAD, success=True, skipped=True)

    # Discover individual renders
    render_mp4s = sorted(work_dir.glob("*_render.mp4"))
    if not render_mp4s:
        # Fall back to output.mp4 if no individual renders
        output_mp4 = work_dir / "output.mp4"
        if output_mp4.exists():
            render_mp4s = [output_mp4]
        else:
            return StageResult(stage=Stage.UPLOAD, success=False,
                               error="No video files to upload")

    highlights_mp4 = work_dir / "highlights.mp4"

    # Load or generate manifest
    manifest_path = match_dir / "manifest.json"
    if manifest_path.exists():
        from wpv.ingest.manifest import load_manifest
        manifest = load_manifest(manifest_path)
    else:
        from wpv.ingest.manifest import generate_manifest
        manifest = generate_manifest(match_dir)
        manifest_path.write_text(manifest.model_dump_json(indent=2))

    from wpv.publish.youtube import (
        build_metadata_from_manifest,
        get_authenticated_service,
        upload_video,
    )

    service = get_authenticated_service()
    base_title = manifest.teams
    num_clips = len(render_mp4s)
    all_video_ids: list[str] = []
    all_urls: list[str] = []

    # Upload each clip render
    for i, rmp4 in enumerate(render_mp4s):
        if num_clips == 1:
            title = base_title
        else:
            title = f"{base_title} - Part {i + 1}"

        meta = build_metadata_from_manifest(manifest, title_override=title)
        result = upload_video(service, rmp4, meta)
        all_video_ids.append(result.video_id)
        all_urls.append(result.url)

    # Upload highlights reel
    if highlights_mp4.exists():
        hl_title = f"{base_title} - Highlights"
        meta = build_metadata_from_manifest(manifest, title_override=hl_title)
        result = upload_video(service, highlights_mp4, meta)
        all_video_ids.append(result.video_id)
        all_urls.append(result.url)

    # Store first video ID for skip-check; store all URLs as JSON
    update_field(match_id, "youtube_video_id", all_video_ids[0], db_path)
    update_field(match_id, "youtube_url", json.dumps(all_urls), db_path)
    update_field(match_id, "uploaded_at", _now(), db_path)

    return StageResult(stage=Stage.UPLOAD, success=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_videos(match_dir: Path) -> list[Path]:
    """Find source .mp4 files in match_dir (excluding work/)."""
    return sorted(
        p for p in match_dir.glob("*.mp4")
        if p.parent == match_dir  # don't recurse into work/
    )


def _video_stem_to_clip_name(stem: str) -> str | None:
    """Derive clip_name from a video filename stem.

    ``PRO_VID_20260131_145519_00_001`` → ``clip_000`` (index 001 → 0-based 000).
    Returns *None* if the stem doesn't match the expected pattern.
    """
    import re

    m = re.search(r"_(\d{3})$", stem)
    if m:
        idx = int(m.group(1))  # 1-based video index
        return f"clip_{idx - 1:03d}"
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
