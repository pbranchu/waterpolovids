"""Tests for wpv.publish.youtube — metadata building + mocked upload."""

from unittest.mock import MagicMock, patch

import pytest

from wpv.ingest.manifest import ClipInfo, MatchManifest
from wpv.publish.youtube import (
    UploadMetadata,
    UploadResult,
    build_metadata_from_manifest,
    upload_from_manifest,
    upload_video,
)


@pytest.fixture()
def manifest() -> MatchManifest:
    return MatchManifest(
        match_id="test_match",
        date="2025-01-15",
        time="14:00:00",
        teams="Tigers vs Sharks",
        location="Pool Arena",
        clips=[
            ClipInfo(
                filename="PRO_VID_20250115_140000_00_001.mp4",
                sequence=0,
                clip_id=1,
                format="SINGLE_LENS",
                width=3840,
                height=2160,
                fps=25.0,
                duration=600.0,
                codec="h264",
                has_lrv=False,
            ),
            ClipInfo(
                filename="PRO_VID_20250115_140000_00_002.mp4",
                sequence=0,
                clip_id=2,
                format="SINGLE_LENS",
                width=3840,
                height=2160,
                fps=25.0,
                duration=600.0,
                codec="h264",
                has_lrv=False,
            ),
        ],
    )


def test_build_metadata_title(manifest: MatchManifest):
    meta = build_metadata_from_manifest(manifest)
    assert meta.title == "Tigers vs Sharks - 2025-01-15 - Pool Arena"


def test_build_metadata_tags(manifest: MatchManifest):
    meta = build_metadata_from_manifest(manifest)
    assert "water polo" in meta.tags
    assert "Tigers" in meta.tags
    assert "Sharks" in meta.tags
    assert "2025-01-15" in meta.tags
    assert "Pool Arena" in meta.tags


def test_build_metadata_description(manifest: MatchManifest):
    meta = build_metadata_from_manifest(manifest)
    assert "Tigers vs Sharks" in meta.description
    assert "2 clip(s)" in meta.description
    assert "20.0 min" in meta.description


def test_build_metadata_privacy_override(manifest: MatchManifest):
    meta = build_metadata_from_manifest(manifest, privacy="private")
    assert meta.privacy == "private"


def test_build_metadata_title_override(manifest: MatchManifest):
    meta = build_metadata_from_manifest(manifest, title_override="Custom Title")
    assert meta.title == "Custom Title"


def test_upload_video_calls_api(tmp_path):
    video = tmp_path / "test.mp4"
    video.write_bytes(b"\x00" * 100)

    meta = UploadMetadata(
        title="Test", description="Desc", tags=["t"], privacy="unlisted",
    )

    mock_service = MagicMock()
    mock_request = MagicMock()
    mock_request.next_chunk.return_value = (None, {"id": "vid123"})
    mock_service.videos.return_value.insert.return_value = mock_request

    with patch("googleapiclient.http.MediaFileUpload") as mock_mfu:
        mock_mfu.return_value = MagicMock()
        result = upload_video(mock_service, video, meta)

    assert isinstance(result, UploadResult)
    assert result.video_id == "vid123"
    assert "vid123" in result.url
