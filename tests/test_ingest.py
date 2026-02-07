"""Tests for the ingest pipeline: format detection, manifest, and stitch."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from wpv.ingest.detect_format import VideoFormat, detect_format, find_lrv
from wpv.ingest.manifest import generate_manifest, parse_insta360_filename
from wpv.ingest.stitch import prepare_equirect

# Use the smallest clip for integration tests
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SMALL_CLIP_DIR = DATA_DIR / "PRO_VID_20260131_153246_00_004-Original"
SMALL_CLIP = SMALL_CLIP_DIR / "PRO_VID_20260131_153246_00_004.mp4"
SMALL_LRV = SMALL_CLIP_DIR / "PRO_LRV_20260131_153246_01_004.lrv"


# --- parse_insta360_filename ---


class TestParseFilename:
    def test_standard_pattern(self):
        result = parse_insta360_filename("PRO_VID_20260131_153246_00_004.mp4")
        assert result["date"] == "2026-01-31"
        assert result["time"] == "15:32:46"
        assert result["sequence"] == 0
        assert result["clip_id"] == 4
        assert result["datetime"] == datetime(2026, 1, 31, 15, 32, 46)

    def test_different_sequence_and_clip(self):
        result = parse_insta360_filename("PRO_VID_20260201_131152_00_011.mp4")
        assert result["date"] == "2026-02-01"
        assert result["time"] == "13:11:52"
        assert result["sequence"] == 0
        assert result["clip_id"] == 11

    def test_embedded_in_path(self):
        result = parse_insta360_filename(
            "/some/path/PRO_VID_20260131_145519_00_001-Original/PRO_VID_20260131_145519_00_001.mp4"
        )
        assert result["clip_id"] == 1

    def test_invalid_pattern_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            parse_insta360_filename("random_video.mp4")


# --- detect_format (integration, requires ffprobe + real data) ---


@pytest.mark.skipif(not SMALL_CLIP.exists(), reason="Test data not available")
class TestDetectFormat:
    def test_equirectangular_detection(self):
        info = detect_format(SMALL_CLIP)
        assert info.format == VideoFormat.EQUIRECTANGULAR
        assert info.width == 4608
        assert info.height == 4608
        assert info.fps == 25.0
        assert info.duration > 0
        assert info.codec == "hevc"

    def test_has_lrv(self):
        info = detect_format(SMALL_CLIP)
        assert info.has_lrv is True
        assert info.lrv_path == SMALL_LRV

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            detect_format(Path("/nonexistent/video.mp4"))


# --- find_lrv ---


@pytest.mark.skipif(not SMALL_CLIP.exists(), reason="Test data not available")
class TestFindLrv:
    def test_finds_matching_lrv(self):
        result = find_lrv(SMALL_CLIP)
        assert result is not None
        assert result.name == "PRO_LRV_20260131_153246_01_004.lrv"

    def test_non_pro_vid_returns_none(self):
        assert find_lrv(Path("/some/random_video.mp4")) is None


# --- manifest generation ---


@pytest.mark.skipif(not SMALL_CLIP.exists(), reason="Test data not available")
class TestManifest:
    def test_generate_produces_valid_json(self):
        m = generate_manifest(SMALL_CLIP_DIR)
        data = json.loads(m.model_dump_json())
        assert data["match_id"] == SMALL_CLIP_DIR.name
        assert data["date"] == "2026-01-31"
        assert len(data["clips"]) == 1
        clip = data["clips"][0]
        assert clip["format"] == "EQUIRECTANGULAR"
        assert clip["width"] == 4608
        assert clip["has_lrv"] is True

    def test_override_teams(self):
        m = generate_manifest(SMALL_CLIP_DIR, teams="Team A vs Team B")
        assert m.teams == "Team A vs Team B"


# --- prepare_equirect ---


@pytest.mark.skipif(not SMALL_CLIP.exists(), reason="Test data not available")
class TestPrepareEquirect:
    def test_symlinks_equirect(self, tmp_path):
        info = detect_format(SMALL_CLIP)
        result = prepare_equirect(info, tmp_path / "work")
        assert result.is_symlink()
        assert result.resolve() == SMALL_CLIP.resolve()

    def test_symlinks_lrv(self, tmp_path):
        info = detect_format(SMALL_CLIP)
        prepare_equirect(info, tmp_path / "work")
        lrv_dest = tmp_path / "work" / SMALL_LRV.name
        assert lrv_dest.is_symlink()
        assert lrv_dest.resolve() == SMALL_LRV.resolve()

    def test_idempotent(self, tmp_path):
        info = detect_format(SMALL_CLIP)
        work = tmp_path / "work"
        prepare_equirect(info, work)
        result = prepare_equirect(info, work)
        assert result.is_symlink()
