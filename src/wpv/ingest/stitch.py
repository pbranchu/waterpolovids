"""Stitch/decode Insta360 .insv files to equirectangular master via Media SDK."""

from __future__ import annotations

import shutil
from pathlib import Path

from wpv.ingest.detect_format import VideoFormat, VideoInfo


def prepare_equirect(video_info: VideoInfo, work_dir: Path | str) -> Path:
    """Prepare an equirectangular video in the work directory.

    - EQUIRECTANGULAR: symlink to work dir (already stitched)
    - DUAL_FISHEYE: would call Insta360 SDK (not yet available)
    - SINGLE_LENS: copy as-is (different render path later)

    Returns the path to the equirectangular file in work_dir.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    dest = work_dir / video_info.path.name

    if dest.exists() or dest.is_symlink():
        dest.unlink()

    if video_info.format == VideoFormat.EQUIRECTANGULAR:
        dest.symlink_to(video_info.path.resolve())
    elif video_info.format == VideoFormat.DUAL_FISHEYE:
        raise NotImplementedError(
            "Dual fisheye stitching requires the Insta360 Media SDK, which is not yet integrated."
        )
    elif video_info.format == VideoFormat.SINGLE_LENS:
        shutil.copy2(video_info.path, dest)

    # Also symlink LRV if available
    if video_info.lrv_path and video_info.lrv_path.exists():
        lrv_dest = work_dir / video_info.lrv_path.name
        if lrv_dest.exists() or lrv_dest.is_symlink():
            lrv_dest.unlink()
        lrv_dest.symlink_to(video_info.lrv_path.resolve())

    return dest
