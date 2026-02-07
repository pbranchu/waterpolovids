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

    # Ball tracking
    hsv_yellow_low: tuple[int, int, int] = (20, 100, 100)
    hsv_yellow_high: tuple[int, int, int] = (35, 255, 255)
    min_ball_px: int = 4
    max_ball_px: int = 60
    detection_confidence_threshold: float = 0.6

    # Tracking state machine
    track_loss_frames: int = 22  # ~0.75s at 30fps
    search_forward_step_s: float = 5.0
    search_max_gap_s: float = 45.0
    rewind_coarse_step_s: float = 0.25

    # Virtual camera
    max_angular_velocity_deg_s: float = 120.0
    default_hfov_deg: float = 70.0
    zoom_hfov_deg: float = 45.0

    # Render
    output_width: int = 1920
    output_height: int = 1080
    output_codec: str = "libx264"
    output_crf: int = 18
    output_fps: float = 30.0

    # Highlights
    highlight_min_duration_s: float = 3.0
    highlight_max_duration_s: float = 480.0  # 8 min

    # YouTube
    youtube_client_secrets: Path = Path("client_secrets.json")


settings = Settings()
