"""JSON serialization for TrackResult."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from wpv.tracking.tracker import TrackGap, TrackPoint, TrackResult


def save_track(result: TrackResult, path: str | Path) -> None:
    """Save a TrackResult to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "clip_name": result.clip_name,
        "video_path": result.video_path,
        "fps": result.fps,
        "frame_count": result.frame_count,
        "width": result.width,
        "height": result.height,
        "detection_scale": result.detection_scale,
        "elapsed_s": result.elapsed_s,
        "stats": result.stats,
        "gaps": [asdict(g) for g in result.gaps],
        "points": [asdict(p) for p in result.points],
    }
    path.write_text(json.dumps(data, indent=2))


def load_track(path: str | Path) -> TrackResult:
    """Load a TrackResult from JSON."""
    data = json.loads(Path(path).read_text())

    points = [
        TrackPoint(
            frame=p["frame"],
            x=p["x"],
            y=p["y"],
            confidence=p["confidence"],
            state=p["state"],
        )
        for p in data["points"]
    ]

    gaps = [
        TrackGap(
            start_frame=g["start_frame"],
            end_frame=g["end_frame"],
            gap_type=g["gap_type"],
        )
        for g in data["gaps"]
    ]

    return TrackResult(
        clip_name=data["clip_name"],
        video_path=data["video_path"],
        fps=data["fps"],
        frame_count=data["frame_count"],
        width=data["width"],
        height=data["height"],
        detection_scale=data["detection_scale"],
        points=points,
        gaps=gaps,
        stats=data.get("stats", {}),
        elapsed_s=data.get("elapsed_s", 0.0),
    )
