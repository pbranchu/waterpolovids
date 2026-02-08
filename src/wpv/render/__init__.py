"""Render reframed video from source footage + tracking data."""

from wpv.render.highlights import (
    HighlightSegment,
    ScoredMoment,
    build_montage,
    load_segments,
    save_segments,
    score_track,
    select_segments,
)
from wpv.render.reframe import (
    CropPath,
    CropRenderer,
    FFmpegWriter,
    PoolBounds,
    compute_crop_path,
    detect_pool_bounds,
    load_game_mask_bounds,
)

__all__ = [
    "CropPath",
    "CropRenderer",
    "FFmpegWriter",
    "HighlightSegment",
    "PoolBounds",
    "ScoredMoment",
    "build_montage",
    "compute_crop_path",
    "detect_pool_bounds",
    "load_game_mask_bounds",
    "load_segments",
    "save_segments",
    "score_track",
    "select_segments",
]
