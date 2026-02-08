"""YouTube publishing via Data API v3."""

from wpv.publish.youtube import (
    UploadMetadata,
    UploadResult,
    add_video_to_playlist,
    build_metadata_from_manifest,
    create_playlist,
    find_playlist,
    get_authenticated_service,
    get_or_create_playlist,
    upload_from_manifest,
    upload_video,
)

__all__ = [
    "UploadMetadata",
    "UploadResult",
    "add_video_to_playlist",
    "build_metadata_from_manifest",
    "create_playlist",
    "find_playlist",
    "get_authenticated_service",
    "get_or_create_playlist",
    "upload_from_manifest",
    "upload_video",
]
