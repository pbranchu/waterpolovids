"""Detect Insta360 file format and validate raw recordings."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class VideoFormat(Enum):
    DUAL_FISHEYE = "DUAL_FISHEYE"
    EQUIRECTANGULAR = "EQUIRECTANGULAR"
    SINGLE_LENS = "SINGLE_LENS"


@dataclass
class VideoInfo:
    path: Path
    format: VideoFormat
    width: int
    height: int
    fps: float
    duration: float
    codec: str
    has_lrv: bool
    lrv_path: Path | None


def _run_ffprobe(path: Path) -> dict:
    """Run ffprobe and return parsed JSON."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def find_lrv(path: Path) -> Path | None:
    """Locate matching LRV preview file for a PRO_VID mp4.

    Swaps PRO_VID→PRO_LRV, _00_→_01_, .mp4→.lrv in the filename.
    """
    name = path.name
    if not name.startswith("PRO_VID_"):
        return None
    lrv_name = name.replace("PRO_VID_", "PRO_LRV_").replace("_00_", "_01_")
    lrv_name = lrv_name.rsplit(".", 1)[0] + ".lrv"
    lrv_path = path.parent / lrv_name
    return lrv_path if lrv_path.exists() else None


def detect_format(path: Path | str) -> VideoInfo:
    """Detect the video format of an Insta360 file using ffprobe."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    probe = _run_ffprobe(path)

    # Find video stream
    video_stream = None
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break
    if video_stream is None:
        raise ValueError(f"No video stream found in {path}")

    width = video_stream["width"]
    height = video_stream["height"]
    codec = video_stream["codec_name"]

    # Parse fps from r_frame_rate (e.g. "25/1")
    fps_parts = video_stream.get("r_frame_rate", "0/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])

    # Duration from format or stream
    duration = float(
        probe.get("format", {}).get("duration", 0)
        or video_stream.get("duration", 0)
    )

    # Classify format
    handler_name = video_stream.get("tags", {}).get("handler_name", "")
    is_square = abs(width - height) < 2  # 1:1 aspect ratio
    aspect = width / height if height else 0

    if is_square and "INS" in handler_name:
        fmt = VideoFormat.EQUIRECTANGULAR
    elif is_square and path.suffix.lower() == ".insv":
        fmt = VideoFormat.DUAL_FISHEYE
    elif abs(aspect - 2.0) < 0.1:
        fmt = VideoFormat.EQUIRECTANGULAR
    else:
        fmt = VideoFormat.SINGLE_LENS

    lrv_path = find_lrv(path)

    return VideoInfo(
        path=path,
        format=fmt,
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        codec=codec,
        has_lrv=lrv_path is not None,
        lrv_path=lrv_path,
    )
