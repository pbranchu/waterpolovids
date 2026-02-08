"""Tests for wpv.db — SQLite results database."""

from pathlib import Path

import pytest

from wpv.db import (
    MatchRecord,
    get_all_records,
    get_record,
    init_db,
    update_field,
    update_status,
    upsert_record,
)


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    p = tmp_path / "test.db"
    init_db(p)
    return p


def _make_record(**overrides) -> MatchRecord:
    defaults = {"match_id": "m1", "source_path": "/data/m1"}
    defaults.update(overrides)
    return MatchRecord(**defaults)


def test_init_creates_db(db_path: Path):
    assert db_path.exists()


def test_upsert_and_get(db_path: Path):
    rec = _make_record()
    upsert_record(rec, db_path)
    fetched = get_record("m1", db_path)
    assert fetched is not None
    assert fetched.match_id == "m1"
    assert fetched.source_path == "/data/m1"
    assert fetched.status == "pending"


def test_get_missing_returns_none(db_path: Path):
    assert get_record("nonexistent", db_path) is None


def test_upsert_replaces(db_path: Path):
    upsert_record(_make_record(status="pending"), db_path)
    upsert_record(_make_record(status="completed"), db_path)
    rec = get_record("m1", db_path)
    assert rec.status == "completed"


def test_get_all_records(db_path: Path):
    upsert_record(_make_record(match_id="a"), db_path)
    upsert_record(_make_record(match_id="b"), db_path)
    upsert_record(_make_record(match_id="c"), db_path)
    records = get_all_records(db_path)
    assert len(records) == 3
    assert [r.match_id for r in records] == ["a", "b", "c"]


def test_update_status(db_path: Path):
    upsert_record(_make_record(), db_path)
    update_status("m1", "failed", error_message="boom", db_path=db_path)
    rec = get_record("m1", db_path)
    assert rec.status == "failed"
    assert rec.error_message == "boom"


def test_update_field(db_path: Path):
    upsert_record(_make_record(), db_path)
    update_field("m1", "youtube_video_id", "abc123", db_path)
    rec = get_record("m1", db_path)
    assert rec.youtube_video_id == "abc123"


def test_update_field_quality_passed_bool(db_path: Path):
    upsert_record(_make_record(), db_path)
    update_field("m1", "quality_passed", True, db_path)
    rec = get_record("m1", db_path)
    assert rec.quality_passed is True


def test_update_field_invalid_raises(db_path: Path):
    upsert_record(_make_record(), db_path)
    with pytest.raises(ValueError, match="Invalid field"):
        update_field("m1", "match_id", "x", db_path)


def test_update_field_coverage(db_path: Path):
    upsert_record(_make_record(), db_path)
    update_field("m1", "track_coverage_pct", 85.5, db_path)
    rec = get_record("m1", db_path)
    assert rec.track_coverage_pct == pytest.approx(85.5)
