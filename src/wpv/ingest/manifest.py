"""Parse and validate manifest.json for a match."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from wpv.ingest.detect_format import VideoFormat, detect_format


class MatchManifest(BaseModel):
    match_id: str = ""
    date: str = ""
    time: str = ""
    teams: str = "unknown"
    location: str = "unknown"
    clips: list[ClipInfo] = Field(default_factory=list)


class ClipInfo(BaseModel):
    filename: str
    sequence: int = 0
    clip_id: int = 0
    format: str = "single_lens"
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration: float = 0.0
    codec: str = "unknown"
    has_lrv: bool = False
    lrv_filename: str | None = None


# Fix forward reference
MatchManifest.model_rebuild()

# Pattern: PRO_VID_YYYYMMDD_HHMMSS_NN_NNN
_FILENAME_RE = re.compile(
    r"PRO_VID_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_(\d{2})_(\d{3})"
)


def parse_insta360_filename(name: str) -> dict[str, Any]:
    """Extract date, time, sequence, clip_id from PRO_VID filename pattern."""
    m = _FILENAME_RE.search(name)
    if not m:
        raise ValueError(f"Filename does not match PRO_VID pattern: {name}")
    year, month, day, hour, minute, second, seq, clip_id = m.groups()
    return {
        "date": f"{year}-{month}-{day}",
        "time": f"{hour}:{minute}:{second}",
        "sequence": int(seq),
        "clip_id": int(clip_id),
        "datetime": datetime(int(year), int(month), int(day), int(hour), int(minute), int(second)),
    }


def generate_manifest(video_path: Path | str, **overrides: Any) -> MatchManifest:
    """Generate a manifest from a directory of video files or a single video.

    Scans for PRO_VID_*.mp4 files, probes each, and builds a MatchManifest.
    Override teams, location, or match_id via keyword arguments.
    """
    video_path = Path(video_path)

    if video_path.is_file():
        video_dir = video_path.parent
        videos = [video_path]
    else:
        video_dir = video_path
        videos = sorted(video_dir.glob("PRO_VID_*.mp4"))

    if not videos:
        raise FileNotFoundError(f"No PRO_VID_*.mp4 files found in {video_dir}")

    # Parse first filename for match-level metadata
    first_info = parse_insta360_filename(videos[0].name)
    match_id = overrides.pop("match_id", video_dir.name)

    clips = []
    for vp in videos:
        file_info = parse_insta360_filename(vp.name)
        vi = detect_format(vp)
        clips.append(ClipInfo(
            filename=vp.name,
            sequence=file_info["sequence"],
            clip_id=file_info["clip_id"],
            format=vi.format.value,
            width=vi.width,
            height=vi.height,
            fps=vi.fps,
            duration=vi.duration,
            codec=vi.codec,
            has_lrv=vi.has_lrv,
            lrv_filename=vi.lrv_path.name if vi.lrv_path else None,
        ))

    return MatchManifest(
        match_id=match_id,
        date=overrides.pop("date", first_info["date"]),
        time=overrides.pop("time", first_info["time"]),
        teams=overrides.pop("teams", "unknown"),
        location=overrides.pop("location", "unknown"),
        clips=clips,
    )


def load_manifest(path: Path | str) -> MatchManifest:
    """Read an existing manifest.json file."""
    path = Path(path)
    return MatchManifest.model_validate_json(path.read_text())
