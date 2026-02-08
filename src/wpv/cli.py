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
def render(match_id: str = typer.Argument(..., help="Match identifier")):
    """Render reframed video from equirectangular master + camera path."""
    typer.echo(f"[render] {match_id} — not yet implemented")
    raise typer.Exit(1)


@app.command()
def highlights(match_id: str = typer.Argument(..., help="Match identifier")):
    """Extract highlight montage."""
    typer.echo(f"[highlights] {match_id} — not yet implemented")
    raise typer.Exit(1)


@app.command()
def upscale(match_id: str = typer.Argument(..., help="Match identifier")):
    """Apply AI upscaling to rendered output."""
    typer.echo(f"[upscale] {match_id} — not yet implemented")
    raise typer.Exit(1)


@app.command()
def publish(match_id: str = typer.Argument(..., help="Match identifier")):
    """Upload to YouTube."""
    typer.echo(f"[publish] {match_id} — not yet implemented")
    raise typer.Exit(1)


@app.command()
def run_all(match_id: str = typer.Argument(..., help="Match identifier")):
    """Run the full pipeline end-to-end."""
    typer.echo(f"[run-all] {match_id} — not yet implemented")
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
