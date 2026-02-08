"""Tests for wpv.pipeline — orchestrator with mocked stages."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from wpv.db import MatchRecord, get_record, init_db, upsert_record
from wpv.pipeline import (
    ALL_STAGES,
    PipelineResult,
    Stage,
    StageResult,
    run_pipeline,
)


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    p = tmp_path / "pipeline_test.db"
    init_db(p)
    return p


@pytest.fixture()
def match_dir(tmp_path: Path) -> Path:
    d = tmp_path / "test_match"
    d.mkdir()
    # Create a dummy video
    (d / "clip.mp4").write_bytes(b"\x00" * 100)
    return d


def _mock_stage_success(stage: Stage, output: str | None = None):
    """Return a function that returns a successful StageResult."""
    def _fn(match_dir, work_dir, match_id, db_path):
        return StageResult(
            stage=stage,
            success=True,
            output_path=Path(output) if output else None,
        )
    return _fn


def _mock_stage_fail(stage: Stage, error: str = "mock failure"):
    def _fn(match_dir, work_dir, match_id, db_path):
        return StageResult(stage=stage, success=False, error=error)
    return _fn


def test_all_stages_pass(match_dir: Path, db_path: Path):
    """Pipeline succeeds when all stages pass."""
    mocks = {}
    for stage in ALL_STAGES:
        mocks[f"wpv.pipeline._stage_{stage.value.replace('-', '_')}"] = _mock_stage_success(stage)

    with patch.multiple("wpv.pipeline", **{
        "_stage_detect": _mock_stage_success(Stage.DETECT),
        "_stage_decode": _mock_stage_success(Stage.DECODE),
        "_stage_track": _mock_stage_success(Stage.TRACK),
        "_stage_render": _mock_stage_success(Stage.RENDER),
        "_stage_highlights": _mock_stage_success(Stage.HIGHLIGHTS),
        "_stage_quality_check": _mock_stage_success(Stage.QUALITY_CHECK),
        "_stage_upload": _mock_stage_success(Stage.UPLOAD),
    }):
        result = run_pipeline(match_dir, db_path=db_path)

    assert result.success is True
    assert len(result.stages) == len(ALL_STAGES)
    rec = get_record(match_dir.name, db_path)
    assert rec.status == "completed"


def test_fail_fast_on_stage_error(match_dir: Path, db_path: Path):
    """Pipeline stops at first failed stage."""
    with patch.multiple("wpv.pipeline", **{
        "_stage_detect": _mock_stage_success(Stage.DETECT),
        "_stage_decode": _mock_stage_fail(Stage.DECODE, error="decode boom"),
        "_stage_track": _mock_stage_success(Stage.TRACK),
        "_stage_render": _mock_stage_success(Stage.RENDER),
        "_stage_highlights": _mock_stage_success(Stage.HIGHLIGHTS),
        "_stage_quality_check": _mock_stage_success(Stage.QUALITY_CHECK),
        "_stage_upload": _mock_stage_success(Stage.UPLOAD),
    }):
        result = run_pipeline(match_dir, db_path=db_path)

    assert result.success is False
    # Only detect + decode ran (decode failed)
    assert len(result.stages) == 2
    assert result.stages[1].error == "decode boom"
    rec = get_record(match_dir.name, db_path)
    assert rec.status == "failed"
    assert rec.error_message == "decode boom"


def test_skip_upload(match_dir: Path, db_path: Path):
    """skip_upload=True excludes upload stage."""
    with patch.multiple("wpv.pipeline", **{
        "_stage_detect": _mock_stage_success(Stage.DETECT),
        "_stage_decode": _mock_stage_success(Stage.DECODE),
        "_stage_track": _mock_stage_success(Stage.TRACK),
        "_stage_render": _mock_stage_success(Stage.RENDER),
        "_stage_highlights": _mock_stage_success(Stage.HIGHLIGHTS),
        "_stage_quality_check": _mock_stage_success(Stage.QUALITY_CHECK),
        "_stage_upload": _mock_stage_success(Stage.UPLOAD),
    }):
        result = run_pipeline(match_dir, skip_upload=True, db_path=db_path)

    assert result.success is True
    stage_names = [s.stage for s in result.stages]
    assert Stage.UPLOAD not in stage_names


def test_selective_stages(match_dir: Path, db_path: Path):
    """Running only specific stages."""
    with patch.multiple("wpv.pipeline", **{
        "_stage_detect": _mock_stage_success(Stage.DETECT),
        "_stage_decode": _mock_stage_success(Stage.DECODE),
        "_stage_track": _mock_stage_success(Stage.TRACK),
        "_stage_render": _mock_stage_success(Stage.RENDER),
        "_stage_highlights": _mock_stage_success(Stage.HIGHLIGHTS),
        "_stage_quality_check": _mock_stage_success(Stage.QUALITY_CHECK),
        "_stage_upload": _mock_stage_success(Stage.UPLOAD),
    }):
        result = run_pipeline(
            match_dir,
            stages=[Stage.DETECT, Stage.DECODE],
            db_path=db_path,
        )

    assert result.success is True
    assert len(result.stages) == 2


def test_progress_callback(match_dir: Path, db_path: Path):
    """Progress callback is invoked per stage."""
    calls = []

    def cb(stage, msg):
        calls.append((stage, msg))

    with patch.multiple("wpv.pipeline", **{
        "_stage_detect": _mock_stage_success(Stage.DETECT),
        "_stage_decode": _mock_stage_success(Stage.DECODE),
        "_stage_track": _mock_stage_success(Stage.TRACK),
        "_stage_render": _mock_stage_success(Stage.RENDER),
        "_stage_highlights": _mock_stage_success(Stage.HIGHLIGHTS),
        "_stage_quality_check": _mock_stage_success(Stage.QUALITY_CHECK),
        "_stage_upload": _mock_stage_success(Stage.UPLOAD),
    }):
        run_pipeline(match_dir, db_path=db_path, progress_callback=cb)

    # 2 calls per stage (starting + done)
    assert len(calls) == len(ALL_STAGES) * 2


def test_exception_in_stage_caught(match_dir: Path, db_path: Path):
    """Exception raised inside a stage is caught and recorded."""
    def _explode(match_dir, work_dir, match_id, db_path):
        raise RuntimeError("unexpected error")

    with patch.multiple("wpv.pipeline", **{
        "_stage_detect": _explode,
        "_stage_decode": _mock_stage_success(Stage.DECODE),
        "_stage_track": _mock_stage_success(Stage.TRACK),
        "_stage_render": _mock_stage_success(Stage.RENDER),
        "_stage_highlights": _mock_stage_success(Stage.HIGHLIGHTS),
        "_stage_quality_check": _mock_stage_success(Stage.QUALITY_CHECK),
        "_stage_upload": _mock_stage_success(Stage.UPLOAD),
    }):
        result = run_pipeline(match_dir, db_path=db_path)

    assert result.success is False
    assert "unexpected error" in result.stages[0].error
