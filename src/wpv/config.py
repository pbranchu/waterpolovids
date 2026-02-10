"""Pipeline configuration via Pydantic settings."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global pipeline settings, loaded from env vars prefixed WPV_."""

    model_config = {"env_prefix": "WPV_"}

    # Paths
    raw_root: Path = Path("/data/raw")
    work_root: Path = Path("/data/work")
    output_root: Path = Path("/data/output")

    # Stitch / decode
    insta360_sdk_path: Path = Path("/opt/insta360")

    # Ball detection models
    ball_model_path: Path = Path("data/labeling/models/ball_classifier.pth")
    ball_ref_image_path: Path = Path("data/ball_reference.webp")

    # Ball tracking
    hsv_yellow_low: tuple[int, int, int] = (20, 100, 100)
    hsv_yellow_high: tuple[int, int, int] = (35, 255, 255)
    min_ball_px: int = 4
    max_ball_px: int = 60
    detection_confidence_threshold: float = 0.6

    # Tracking state machine
    track_loss_frames: int = 19  # ~0.75s at 25fps
    track_detection_scale: float = 0.5
    track_search_window_frames: int = 125  # 5s at 25fps (search_forward_step)
    track_reacquire_persistence: int = 5
    track_gate_distance: float = 5.0
    search_forward_step_s: float = 5.0
    search_max_gap_s: float = 45.0
    rewind_coarse_step_s: float = 0.25
    track_parallel_clips: int = 4
    parallel_games: int = 2

    # Virtual camera
    max_angular_velocity_deg_s: float = 120.0
    default_hfov_deg: float = 70.0
    zoom_hfov_deg: float = 45.0

    # Render (equirect reframe — future)
    output_width: int = 1920
    output_height: int = 1080
    output_codec: str = "libx264"
    output_crf: int = 18
    output_fps: float = 30.0

    # Render (single-lens crop/pan)
    crop_output_width: int = 1280
    crop_output_height: int = 720
    crop_output_codec: str = "hevc_nvenc"
    crop_output_crf: int = 20
    crop_output_preset: str = "p4"
    crop_smoothing_alpha: float = 0.08
    crop_dead_zone_px: int = 20
    crop_max_velocity_px: float = 120.0

    # Fisheye lens intrinsics (Insta360 X5 single-lens)
    fisheye_focal_length_px: float = 1466.0   # f = 2304 / (pi/2)
    fisheye_center_x: float = 2304.0
    fisheye_center_y: float = 2304.0
    fisheye_undistort: bool = True

    # Highlights
    highlight_min_duration_s: float = 3.0
    highlight_max_duration_s: float = 480.0  # 8 min
    highlight_target_duration_s: float = 300.0
    highlight_score_threshold: float = 0.4
    highlight_context_s: float = 3.0
    highlight_crossfade_s: float = 0.5
    highlight_speed_sigma: float = 2.0
    highlight_direction_window_s: float = 0.5
    highlight_gap_reappear_bonus: float = 0.3
    highlight_max_segments: int = 50

    # Quality gates
    quality_min_track_coverage_pct: float = 60.0
    quality_min_highlight_duration_s: float = 10.0

    # YouTube
    youtube_client_secrets: Path = Path("client_secrets.json")
    youtube_token_path: Path = Path("~/.wpv/youtube_token.json")
    youtube_default_privacy: str = "unlisted"
    youtube_category_id: str = "17"

    # Game masks & shared output
    game_masks_path: Path = Path("data/labeling/game_masks.json")
    shared_output_dir: Path = Path("/mnt/work/shared")

    # Results DB
    results_db_path: Path = Path("~/.wpv/results.db")

    # Web UI
    web_upload_chunk_size_mb: int = 10
    web_port: int = 5000


settings = Settings()
