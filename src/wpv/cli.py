"""CLI entry point for the wpv pipeline."""

from pathlib import Path

import typer

app = typer.Typer(name="wpv", help="Water Polo Video auto-reframe pipeline.")


@app.command()
def detect(video_path: Path = typer.Argument(..., help="Path to a video file")):
    """Detect video format and print info."""
    from wpv.ingest.detect_format import detect_format

    info = detect_format(video_path)
    typer.echo(f"Format:   {info.format.value}")
    typer.echo(f"Size:     {info.width}x{info.height}")
    typer.echo(f"FPS:      {info.fps}")
    typer.echo(f"Duration: {info.duration:.2f}s")
    typer.echo(f"Codec:    {info.codec}")
    typer.echo(f"LRV:      {info.lrv_path or 'not found'}")


@app.command()
def decode(
    video_path: Path = typer.Argument(..., help="Path to a video file"),
    out_dir: Path = typer.Option(..., "-o", "--out-dir", help="Output work directory"),
):
    """Prepare equirectangular video (symlink/stitch/copy)."""
    from wpv.ingest.detect_format import detect_format
    from wpv.ingest.stitch import prepare_equirect

    info = detect_format(video_path)
    result = prepare_equirect(info, out_dir)
    typer.echo(f"[decode] {info.format.value} → {result}")


@app.command()
def manifest(
    video_dir: Path = typer.Argument(..., help="Directory containing PRO_VID_*.mp4 files"),
    out_path: Path = typer.Option(None, "-o", "--out", help="Output manifest.json path"),
    teams: str = typer.Option("unknown", help="Team names"),
    location: str = typer.Option("unknown", help="Pool / venue location"),
):
    """Generate manifest.json from video directory."""
    from wpv.ingest.manifest import generate_manifest

    m = generate_manifest(video_dir, teams=teams, location=location)
    json_str = m.model_dump_json(indent=2)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_str)
        typer.echo(f"[manifest] Written to {out_path}")
    else:
        typer.echo(json_str)


@app.command()
def detect_ball(
    video_path: Path = typer.Argument(..., help="Path to a video file (LRV or MP4)"),
    output: Path = typer.Option("detections.json", "-o", "--output", help="Output JSON path"),
    visualize: bool = typer.Option(False, "--visualize", help="Draw bounding boxes on frames"),
    vis_output: Path = typer.Option(None, "--vis-output", help="Annotated video output path"),
):
    """Run ball detection (HSV + CNN) on a video."""
    from wpv.config import settings
    from wpv.tracking.detector import BallDetector

    detector = BallDetector(
        min_area=settings.min_ball_px,
        max_area=settings.max_ball_px,
        confidence_threshold=settings.detection_confidence_threshold,
    )
    detections = detector.process_video(
        video_path,
        output_path=output,
        visualize=visualize,
        vis_output=vis_output,
    )
    total = sum(len(d) for d in detections)
    typer.echo(f"[detect-ball] {len(detections)} frames, {total} total detections → {output}")


@app.command()
def label_frames(
    video_path: Path = typer.Argument(..., help="Path to a video file"),
    count: int = typer.Option(100, "--count", help="Number of frames to extract"),
    output: Path = typer.Option("frames/", "-o", "--output", help="Output directory"),
    strategy: str = typer.Option("uniform", "--strategy", help="uniform, random, or diverse"),
):
    """Extract frames from a video for labeling."""
    from scripts.label_frames import extract_frames, generate_annotations_stub

    frames = extract_frames(video_path, output, count, strategy)
    ann_path = output / "annotations.json"
    generate_annotations_stub(frames, ann_path)
    typer.echo(f"[label-frames] Extracted {len(frames)} frames → {output}")


@app.command()
def label(
    prep_only: bool = typer.Option(False, "--prep-only", help="Only extract frames and run HSV detection"),
    serve_only: bool = typer.Option(False, "--serve-only", help="Only launch the labeling web UI"),
    batch: bool = typer.Option(False, "--batch", help="Run iterative batch labeling workflow"),
    train: bool = typer.Option(False, "--train", help="Tune HSV params and train CNN from annotations"),
    schedule: str = typer.Option(
        "50,50,50,50,50,50", "--schedule",
        help="Comma-separated batch sizes for --batch mode",
    ),
    count: int = typer.Option(500, "--count", help="Target number of frames to extract"),
    port: int = typer.Option(5001, "--port", help="HTTP port for the labeling UI"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing annotations"),
):
    """Label ball candidates for CNN training data.

    Two-step workflow: prep extracts frames + HSV candidates, serve launches the labeling UI.
    By default runs both steps (prep then serve).

    Use --batch for iterative batch labeling with automatic retraining.
    Use --train to tune HSV params and train CNN from existing annotations.
    """
    import argparse
    import importlib.util
    import sys

    # Import from scripts/ which isn't a package — use importlib
    spec = importlib.util.spec_from_file_location(
        "label_ball",
        Path(__file__).resolve().parents[2] / "scripts" / "label_ball.py",
    )
    label_ball = importlib.util.module_from_spec(spec)
    sys.modules["label_ball"] = label_ball
    spec.loader.exec_module(label_ball)

    if train:
        train_args = argparse.Namespace()
        label_ball.cmd_train(train_args)
        return

    if batch:
        batch_args = argparse.Namespace(schedule=schedule, port=port)
        label_ball.cmd_batch(batch_args)
        return

    run_prep = not serve_only
    run_serve = not prep_only

    if run_prep:
        prep_args = argparse.Namespace(count=count, force=force)
        label_ball.cmd_prep(prep_args)

    if run_serve:
        serve_args = argparse.Namespace(port=port)
        label_ball.cmd_serve(serve_args)


@app.command()
def ingest(match_id: str = typer.Argument(..., help="Match identifier")):
    """Validate incoming footage and manifest."""
    typer.echo(f"[ingest] {match_id} — not yet implemented")
    raise typer.Exit(1)


@app.command()
def stitch(match_id: str = typer.Argument(..., help="Match identifier")):
    """Stitch/decode Insta360 files to equirectangular master."""
    typer.echo(f"[stitch] {match_id} — not yet implemented")
    raise typer.Exit(1)


@app.command()
def track(
    video_path: Path = typer.Argument(..., help="Path to a video file"),
    output: Path = typer.Option("track.json", "-o", "--output", help="Output JSON path"),
    clip: str = typer.Option(None, "--clip", help="Clip name (defaults to filename stem)"),
    scale: float = typer.Option(None, "--scale", help="Detection scale (default from config)"),
    ref_image: Path = typer.Option(None, "--ref-image", help="Ball reference image for scoring"),
    model: Path = typer.Option(None, "--model", help="CNN model path"),
):
    """Track ball through a video clip and save track JSON."""
    from wpv.config import settings
    from wpv.tracking.detector import BallDetector
    from wpv.tracking.track_io import save_track
    from wpv.tracking.tracker import BallTracker

    det_scale = scale if scale is not None else settings.track_detection_scale

    detector = BallDetector(
        model_path=str(model) if model else None,
        ref_image_path=str(ref_image) if ref_image else None,
        min_area=settings.min_ball_px,
        max_area=settings.max_ball_px,
        confidence_threshold=settings.detection_confidence_threshold,
    )
    tracker = BallTracker(
        detector=detector,
        detection_scale=det_scale,
        loss_frames=settings.track_loss_frames,
        search_step_s=settings.search_forward_step_s,
        search_max_gap_s=settings.search_max_gap_s,
        rewind_step_s=settings.rewind_coarse_step_s,
        reacquire_persistence=settings.track_reacquire_persistence,
        gate_distance=settings.track_gate_distance,
        confidence_threshold=settings.detection_confidence_threshold,
    )

    def _progress(frame: int, total: int, state: str) -> None:
        typer.echo(f"  [track] {frame}/{total} ({state})")

    result = tracker.track_clip(
        video_path, clip_name=clip, progress_callback=_progress,
    )
    save_track(result, output)
    typer.echo(
        f"[track] {result.clip_name}: {result.stats.get('num_points', 0)} points, "
        f"{result.stats.get('num_gaps', 0)} gaps, {result.elapsed_s:.1f}s → {output}"
    )


@app.command()
def track_all(
    video_dir: Path = typer.Argument(
        ..., help="Directory containing PRO_VID_*.mp4 files"
    ),
    out_dir: Path = typer.Option(
        "data/tracks", "-o", "--out-dir", help="Output directory for track JSONs"
    ),
    scale: float = typer.Option(None, "--scale", help="Detection scale (default from config)"),
    ref_image: Path = typer.Option(None, "--ref-image", help="Ball reference image for scoring"),
    model: Path = typer.Option(None, "--model", help="CNN model path"),
):
    """Track ball in all video clips found in a directory."""
    from wpv.config import settings
    from wpv.tracking.detector import BallDetector
    from wpv.tracking.track_io import save_track
    from wpv.tracking.tracker import BallTracker

    det_scale = scale if scale is not None else settings.track_detection_scale

    videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        typer.echo(f"[track-all] No .mp4 files found in {video_dir}")
        raise typer.Exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    detector = BallDetector(
        model_path=str(model) if model else None,
        ref_image_path=str(ref_image) if ref_image else None,
        min_area=settings.min_ball_px,
        max_area=settings.max_ball_px,
        confidence_threshold=settings.detection_confidence_threshold,
    )
    tracker = BallTracker(
        detector=detector,
        detection_scale=det_scale,
        loss_frames=settings.track_loss_frames,
        search_step_s=settings.search_forward_step_s,
        search_max_gap_s=settings.search_max_gap_s,
        rewind_step_s=settings.rewind_coarse_step_s,
        reacquire_persistence=settings.track_reacquire_persistence,
        gate_distance=settings.track_gate_distance,
        confidence_threshold=settings.detection_confidence_threshold,
    )

    for i, vp in enumerate(videos, 1):
        clip_name = vp.stem
        out_path = out_dir / f"{clip_name}.json"
        typer.echo(f"[track-all] ({i}/{len(videos)}) {clip_name}")

        def _progress(frame: int, total: int, state: str) -> None:
            typer.echo(f"  [track] {frame}/{total} ({state})")

        result = tracker.track_clip(
            vp, clip_name=clip_name, progress_callback=_progress,
        )
        save_track(result, out_path)
        typer.echo(
            f"  → {result.stats.get('num_points', 0)} points, "
            f"{result.stats.get('num_gaps', 0)} gaps, {result.elapsed_s:.1f}s"
        )

    typer.echo(f"[track-all] Done. {len(videos)} clips → {out_dir}")


@app.command()
def camera(match_id: str = typer.Argument(..., help="Match identifier")):
    """Generate virtual camera path from tracking data."""
    typer.echo(f"[camera] {match_id} — not yet implemented")
    raise typer.Exit(1)


@app.command()
def render(
    video_path: Path = typer.Argument(..., help="Path to source video file"),
    track_path: Path = typer.Argument(..., help="Path to track JSON from `wpv track`"),
    output: Path = typer.Option("output.mp4", "-o", "--output", help="Output video path"),
    preview: bool = typer.Option(False, "--preview", help="Fast preview (half-res, crf 28)"),
    width: int = typer.Option(None, "--width", help="Crop width (default from config)"),
    height: int = typer.Option(None, "--height", help="Crop height (default from config)"),
    crf: int = typer.Option(None, "--crf", help="Output CRF (default from config)"),
    game_masks: Path = typer.Option(
        None, "--game-masks", help="Path to game_masks.json with manual pool boundaries"
    ),
    clip_name: str = typer.Option(
        None, "--clip-name", help="Clip key in game_masks.json, e.g. clip_000"
    ),
    start_frame: int = typer.Option(0, "--start-frame", help="First frame to render (0-based)"),
    end_frame: int = typer.Option(0, "--end-frame", help="Last frame to render (0 = end of clip)"),
    no_undistort: bool = typer.Option(False, "--no-undistort", help="Disable fisheye undistortion"),
    hfov: float = typer.Option(None, "--hfov", help="Output horizontal FOV in degrees (default 70)"),
):
    """Render crop-and-pan video following tracked ball positions."""
    from wpv.config import settings
    from wpv.render.reframe import CropRenderer, load_game_mask_bounds
    from wpv.tracking.track_io import load_track

    track = load_track(track_path)
    crop_w = width if width is not None else settings.crop_output_width
    crop_h = height if height is not None else settings.crop_output_height
    out_crf = crf if crf is not None else settings.crop_output_crf

    # Load pool bounds from manual mask or fall back to auto-detection
    pool_bounds = None
    if game_masks and clip_name:
        pool_bounds = load_game_mask_bounds(game_masks, clip_name)
        typer.echo(
            f"[render] Pool bounds from {clip_name}: "
            f"x=[{pool_bounds.x_min}, {pool_bounds.x_max}] "
            f"y=[{pool_bounds.y_min}, {pool_bounds.y_max}]"
        )

    def _progress(done: int, total: int) -> None:
        if done % 250 == 0 or done == total:
            typer.echo(f"  [render] {done}/{total} frames")

    undistort = not no_undistort and settings.fisheye_undistort
    out_hfov = hfov if hfov is not None else settings.default_hfov_deg

    renderer = CropRenderer(
        video_path=video_path,
        track=track,
        output_path=output,
        crop_w=crop_w,
        crop_h=crop_h,
        alpha=settings.crop_smoothing_alpha,
        dead_zone=settings.crop_dead_zone_px,
        max_vel=settings.crop_max_velocity_px,
        codec=settings.crop_output_codec,
        crf=out_crf,
        preset=settings.crop_output_preset,
        preview=preview,
        progress_callback=_progress,
        pool_bounds=pool_bounds,
        start_frame=start_frame,
        end_frame=end_frame,
        undistort=undistort,
        hfov_deg=out_hfov,
    )
    result = renderer.run()
    typer.echo(f"[render] Done → {result}")


@app.command()
def highlights(
    track_path: Path = typer.Argument(..., help="Path to track JSON from `wpv track`"),
    output: Path = typer.Option("segments.json", "-o", "--output", help="Output segments JSON (or .mp4 with --render)"),
    render_video: Path = typer.Option(None, "--render", help="Source video to render montage from"),
    target_duration: float = typer.Option(None, "--target-duration", help="Target highlight duration (s)"),
    threshold: float = typer.Option(None, "--threshold", help="Score threshold for highlights"),
    crossfade: float = typer.Option(None, "--crossfade", help="Crossfade duration between segments (s)"),
    preview: bool = typer.Option(False, "--preview", help="Fast preview (lower quality)"),
):
    """Score track data and extract highlight segments (+ optional montage render)."""
    from wpv.config import settings
    from wpv.render.highlights import (
        build_montage,
        save_segments,
        score_track,
        select_segments,
    )
    from wpv.tracking.track_io import load_track

    track = load_track(track_path)

    tgt_dur = target_duration if target_duration is not None else settings.highlight_target_duration_s
    thr = threshold if threshold is not None else settings.highlight_score_threshold
    xfade = crossfade if crossfade is not None else settings.highlight_crossfade_s

    scores = score_track(
        track,
        speed_sigma_threshold=settings.highlight_speed_sigma,
        direction_window_s=settings.highlight_direction_window_s,
        gap_reappear_bonus=settings.highlight_gap_reappear_bonus,
    )
    segments = select_segments(
        scores,
        fps=track.fps,
        threshold=thr,
        context_s=settings.highlight_context_s,
        min_segment_s=settings.highlight_min_duration_s,
        target_duration_s=tgt_dur,
        max_segments=settings.highlight_max_segments,
    )

    total_dur = sum(s.end_s - s.start_s for s in segments)
    typer.echo(f"[highlights] {len(segments)} segments, {total_dur:.1f}s total")

    if render_video is not None:
        crf = 28 if preview else settings.crop_output_crf
        preset = "ultrafast" if preview else settings.crop_output_preset
        montage_out = output if str(output).endswith(".mp4") else output.with_suffix(".mp4")
        build_montage(
            segments,
            source_video=render_video,
            output_path=montage_out,
            fps=track.fps,
            crossfade_s=xfade,
            codec=settings.crop_output_codec,
            crf=crf,
            preset=preset,
        )
        typer.echo(f"[highlights] Montage → {montage_out}")
    else:
        save_segments(segments, output)
        typer.echo(f"[highlights] Segments → {output}")


@app.command()
def quality_check(
    track_path: Path = typer.Argument(..., help="Path to track JSON from `wpv track`"),
    segments_path: Path = typer.Option(None, "--segments", help="Path to segments JSON from `wpv highlights`"),
    strict: bool = typer.Option(False, "--strict", help="Exit code 1 on gate failure"),
    min_coverage: float = typer.Option(None, "--min-coverage", help="Min track coverage %"),
    min_hl_duration: float = typer.Option(None, "--min-hl-duration", help="Min highlight duration (s)"),
):
    """Run pre-publish quality gates on track (and optional highlight) data."""
    from wpv.config import settings
    from wpv.quality import run_quality_gates
    from wpv.render.highlights import load_segments
    from wpv.tracking.track_io import load_track

    track = load_track(track_path)

    segments = None
    if segments_path is not None:
        segments = load_segments(segments_path)

    cov = min_coverage if min_coverage is not None else settings.quality_min_track_coverage_pct
    hl_dur = min_hl_duration if min_hl_duration is not None else settings.quality_min_highlight_duration_s

    report = run_quality_gates(
        track,
        segments=segments,
        min_track_coverage_pct=cov,
        min_highlight_duration_s=hl_dur,
    )

    typer.echo(f"Track coverage:     {report.track_coverage_pct:.1f}%  {'PASS' if report.track_coverage_ok else 'FAIL'}")
    typer.echo(f"Highlight duration: {report.highlight_duration_s:.1f}s  {'PASS' if report.highlight_duration_ok else 'FAIL'}")
    typer.echo(f"Overall:            {'PASS' if report.passed else 'FAIL'}")

    if strict and not report.passed:
        raise typer.Exit(1)


@app.command()
def upscale(match_id: str = typer.Argument(..., help="Match identifier")):
    """Apply AI upscaling to rendered output."""
    typer.echo(f"[upscale] {match_id} — not yet implemented")
    raise typer.Exit(1)


@app.command()
def process(
    match_dir: Path = typer.Argument(..., help="Directory containing match video files"),
    skip_upload: bool = typer.Option(False, "--skip-upload", help="Skip YouTube upload"),
    stages: str = typer.Option(None, "--stages", help="Comma-separated stages to run (detect,decode,track,render,highlights,quality-check,upload)"),
):
    """Run the full pipeline on a single match directory."""
    from wpv.pipeline import Stage, run_pipeline

    active_stages = None
    if stages:
        active_stages = [Stage(s.strip()) for s in stages.split(",")]

    def _progress(stage: Stage, msg: str) -> None:
        typer.echo(f"  [{stage.value}] {msg}")

    result = run_pipeline(
        match_dir,
        stages=active_stages,
        skip_upload=skip_upload,
        progress_callback=_progress,
    )

    if result.success:
        typer.echo(f"[process] {result.match_id} completed successfully")
        if result.youtube_url:
            typer.echo(f"  YouTube: {result.youtube_url}")
    else:
        failed = [s for s in result.stages if not s.success and not s.skipped]
        if failed:
            typer.echo(f"[process] {result.match_id} failed at {failed[0].stage.value}: {failed[0].error}")
        raise typer.Exit(1)


@app.command()
def batch(
    input_dir: Path = typer.Argument(..., help="Directory containing match subdirectories"),
    resume: bool = typer.Option(False, "--resume", help="Skip already-completed matches"),
    parallel: int = typer.Option(1, "--parallel", help="Number of parallel workers"),
    skip_upload: bool = typer.Option(False, "--skip-upload", help="Skip YouTube upload"),
):
    """Run pipeline on all match directories under input_dir."""
    import concurrent.futures

    from wpv.db import get_record, init_db
    from wpv.pipeline import run_pipeline

    init_db()

    match_dirs = sorted(
        d for d in input_dir.iterdir()
        if d.is_dir() and list(d.glob("*.mp4"))
    )

    if not match_dirs:
        typer.echo(f"[batch] No match directories with .mp4 files found in {input_dir}")
        raise typer.Exit(1)

    if resume:
        to_run = []
        for d in match_dirs:
            rec = get_record(d.name)
            if rec and rec.status == "completed":
                typer.echo(f"  [batch] Skipping {d.name} (completed)")
                continue
            to_run.append(d)
        match_dirs = to_run

    typer.echo(f"[batch] {len(match_dirs)} match(es) to process")

    def _run_one(d: Path) -> tuple[str, bool, str | None]:
        r = run_pipeline(d, skip_upload=skip_upload)
        err = None
        if not r.success:
            failed = [s for s in r.stages if not s.success and not s.skipped]
            err = failed[0].error if failed else "unknown error"
        return r.match_id, r.success, err

    if parallel > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(_run_one, d): d for d in match_dirs}
            for fut in concurrent.futures.as_completed(futures):
                match_id, ok, err = fut.result()
                if ok:
                    typer.echo(f"  [batch] {match_id} — completed")
                else:
                    typer.echo(f"  [batch] {match_id} — FAILED: {err}")
    else:
        for d in match_dirs:
            match_id, ok, err = _run_one(d)
            if ok:
                typer.echo(f"  [batch] {match_id} — completed")
            else:
                typer.echo(f"  [batch] {match_id} — FAILED: {err}")

    typer.echo("[batch] Done")


@app.command()
def upload(
    video: Path = typer.Argument(..., help="Path to video file to upload"),
    manifest_path: Path = typer.Option(..., "-m", "--manifest", help="Path to manifest.json"),
    privacy: str = typer.Option(None, "--privacy", help="Privacy setting (unlisted, private, public)"),
    title: str = typer.Option(None, "--title", help="Override video title"),
):
    """Upload a video to YouTube."""
    from wpv.ingest.manifest import load_manifest
    from wpv.publish.youtube import upload_from_manifest

    m = load_manifest(manifest_path)
    typer.echo(f"[upload] Uploading {video.name}...")
    result = upload_from_manifest(video, m, privacy=privacy, title=title)
    typer.echo(f"[upload] Done — {result.url}")


@app.command()
def ui(
    port: int = typer.Option(5000, "--port", help="HTTP port for the web UI"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),
):
    """Launch the web UI for game setup and processing."""
    from wpv.web.app import create_app

    web_app = create_app()
    typer.echo(f"[ui] Starting web UI on http://{host}:{port}")
    web_app.run(host=host, port=port, threaded=True)


@app.command()
def status(
    match: str = typer.Option(None, "--match", help="Show detail for a specific match ID"),
):
    """Show processing status of all matches."""
    from wpv.db import get_all_records, get_record, init_db

    init_db()

    if match:
        rec = get_record(match)
        if rec is None:
            typer.echo(f"[status] No record found for {match}")
            raise typer.Exit(1)
        typer.echo(f"Match:      {rec.match_id}")
        typer.echo(f"Source:     {rec.source_path}")
        typer.echo(f"Status:     {rec.status}")
        typer.echo(f"Started:    {rec.started_at or '-'}")
        typer.echo(f"Completed:  {rec.completed_at or '-'}")
        typer.echo(f"Error:      {rec.error_message or '-'}")
        typer.echo(f"Coverage:   {rec.track_coverage_pct or '-'}%")
        typer.echo(f"Quality:    {'PASS' if rec.quality_passed else ('FAIL' if rec.quality_passed is not None else '-')}")
        typer.echo(f"YouTube:    {rec.youtube_url or '-'}")
        return

    records = get_all_records()
    if not records:
        typer.echo("[status] No matches recorded yet")
        return

    # Table header
    typer.echo(f"{'MATCH':<30} {'STATUS':<12} {'COVERAGE':<10} {'QUALITY':<8} {'YOUTUBE'}")
    typer.echo("-" * 90)
    for r in records:
        cov = f"{r.track_coverage_pct:.1f}%" if r.track_coverage_pct is not None else "-"
        qual = "PASS" if r.quality_passed else ("FAIL" if r.quality_passed is not None else "-")
        yt = r.youtube_url or "-"
        typer.echo(f"{r.match_id:<30} {r.status:<12} {cov:<10} {qual:<8} {yt}")


if __name__ == "__main__":
    app()
