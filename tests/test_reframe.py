"""Tests for single-lens crop/pan render with dynamic zoom."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from wpv.render.reframe import (
    CropPath, CropRenderer, FFmpegWriter, FisheyeIntrinsics, PoolBounds,
    build_fisheye_to_perspective_maps, compute_crop_path, fisheye_hfov_for_zoom,
    _compute_zoom, _fill_gaps_hold_snap, _remove_aberrations,
    _smooth_zone_aware, load_game_mask_bounds,
)
from wpv.tracking.tracker import TrackPoint, TrackResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_track(
    frame_count: int = 100,
    width: int = 4608,
    height: int = 4608,
    fps: float = 25.0,
    points: list[TrackPoint] | None = None,
) -> TrackResult:
    """Create a minimal TrackResult for testing."""
    return TrackResult(
        clip_name="test",
        video_path="/fake/video.mp4",
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        detection_scale=0.5,
        points=points or [],
        gaps=[],
        stats={},
        elapsed_s=0.0,
    )


def _make_points_at(x: float, y: float, start: int, end: int) -> list[TrackPoint]:
    """Create TrackPoints at a fixed position for a range of frames."""
    return [
        TrackPoint(frame=i, x=x, y=y, confidence=0.9, state="tracking")
        for i in range(start, end)
    ]


# ---------------------------------------------------------------------------
# compute_crop_path — fill pass
# ---------------------------------------------------------------------------


class TestCropPathFill:
    """Pass 1: raw position filling and gap interpolation."""

    def test_no_points_defaults_to_centre(self):
        """Empty track → crop centres on frame centre."""
        track = _make_track(frame_count=10, width=4608, height=4608)
        path = compute_crop_path(track, crop_w=1280, crop_h=720, dead_zone=0, end_zone_frac=0)

        # Centre of a 4608×4608 frame is (2304, 2304).
        assert path.cxs[0] == pytest.approx(2304, abs=1)
        assert path.cys[0] == pytest.approx(2304, abs=1)
        # All frames identical
        np.testing.assert_array_almost_equal(path.cxs, path.cxs[0])
        np.testing.assert_array_almost_equal(path.cys, path.cys[0])

    def test_single_point_fills_all_frames(self):
        """A single TrackPoint fills every frame with the same position."""
        pts = [TrackPoint(frame=5, x=1000.0, y=2000.0, confidence=0.9, state="tracking")]
        track = _make_track(frame_count=10, points=pts)
        path = compute_crop_path(track, crop_w=1280, crop_h=720, dead_zone=0, end_zone_frac=0)

        np.testing.assert_array_almost_equal(path.cxs, 1000, decimal=0)
        np.testing.assert_array_almost_equal(path.cys, 2000, decimal=0)

    def test_gap_interpolation(self):
        """Points at frames 0 and 9 → intermediate frames are linearly filled."""
        pts = [
            TrackPoint(frame=0, x=1000.0, y=2000.0, confidence=0.9, state="tracking"),
            TrackPoint(frame=9, x=2000.0, y=3000.0, confidence=0.9, state="tracking"),
        ]
        track = _make_track(frame_count=10, points=pts)
        # alpha=1.0, max_vel=9999 to disable smoothing for this test
        path = compute_crop_path(
            track, crop_w=1280, crop_h=720, alpha=1.0, dead_zone=0, max_vel=9999,
            end_zone_frac=0,
        )

        # Centres should increase monotonically
        assert path.cxs[-1] > path.cxs[0]
        assert path.cys[-1] > path.cys[0]


# ---------------------------------------------------------------------------
# compute_crop_path — smooth pass
# ---------------------------------------------------------------------------


class TestCropPathSmooth:
    """Pass 2: EMA smoothing, dead zone, velocity clamp."""

    def test_dead_zone_holds_position(self):
        """Small displacement within dead zone → camera doesn't move."""
        pts = _make_points_at(2304, 2304, 0, 50)
        # Frame 50 onward shift by less than dead zone
        pts += _make_points_at(2304 + 10, 2304, 50, 100)
        track = _make_track(frame_count=100, points=pts)
        path = compute_crop_path(track, dead_zone=30, end_zone_frac=0)

        # Camera should not have moved
        np.testing.assert_array_almost_equal(path.cxs, path.cxs[0], decimal=0)

    def test_large_jump_is_smoothed(self):
        """A sudden ball jump produces gradual camera movement, not a snap."""
        pts = _make_points_at(1000, 2304, 0, 50)
        pts += _make_points_at(3000, 2304, 50, 100)
        track = _make_track(frame_count=100, points=pts)
        path = compute_crop_path(track, alpha=0.05, dead_zone=0, max_vel=150, end_zone_frac=0)

        # Frame 51 should NOT have jumped fully to the new position
        assert path.cxs[51] < 3000
        # But by frame 99 it should be closer to the new position
        assert path.cxs[99] > path.cxs[51]

    def test_velocity_clamp(self):
        """Per-frame displacement is clamped to max_vel."""
        pts = _make_points_at(1000, 2304, 0, 1)
        pts += _make_points_at(4000, 2304, 1, 100)
        track = _make_track(frame_count=100, points=pts)
        path = compute_crop_path(track, alpha=1.0, dead_zone=0, max_vel=50, end_zone_frac=0)

        # With alpha=1.0, each frame tries to jump fully but velocity clamp
        # limits to 50px/frame.
        diffs = np.abs(np.diff(path.cxs))
        assert np.all(diffs <= 51)  # small rounding tolerance


# ---------------------------------------------------------------------------
# compute_crop_path — clamp pass
# ---------------------------------------------------------------------------


class TestCropPathClamp:
    """Pass 3: boundary clamping."""

    def test_clamp_left_edge(self):
        """Ball near left edge → crop doesn't go past left frame edge."""
        pts = _make_points_at(100, 2304, 0, 10)
        track = _make_track(frame_count=10, width=4608, height=4608, points=pts)
        path = compute_crop_path(track, crop_w=1280, crop_h=720, dead_zone=0, end_zone_frac=0)

        # Centre must be at least half-crop from left edge (crop left >= 0)
        assert np.all(path.cxs >= 1280 / 2)

    def test_clamp_right_edge(self):
        """Ball near right edge → crop stays within frame."""
        pts = _make_points_at(4500, 2304, 0, 10)
        track = _make_track(frame_count=10, width=4608, height=4608, points=pts)
        path = compute_crop_path(track, crop_w=1280, crop_h=720, dead_zone=0, end_zone_frac=0)

        # Centre must be at most half-crop from right edge (crop right <= 4608)
        assert np.all(path.cxs <= 4608 - 1280 / 2)

    def test_clamp_top_edge(self):
        """Ball near top edge → crop doesn't go past top frame edge."""
        pts = _make_points_at(2304, 50, 0, 10)
        track = _make_track(frame_count=10, points=pts)
        path = compute_crop_path(track, crop_w=1280, crop_h=720, dead_zone=0, end_zone_frac=0)

        assert np.all(path.cys >= 720 / 2)

    def test_clamp_bottom_edge(self):
        """Ball near bottom edge → crop stays within frame."""
        pts = _make_points_at(2304, 4550, 0, 10)
        track = _make_track(frame_count=10, width=4608, height=4608, points=pts)
        path = compute_crop_path(track, crop_w=1280, crop_h=720, dead_zone=0, end_zone_frac=0)

        assert np.all(path.cys <= 4608 - 720 / 2)

    def test_crop_larger_than_source_clamps_reasonably(self):
        """When crop >= source, centres stay within source bounds."""
        pts = _make_points_at(50, 50, 0, 5)
        track = _make_track(frame_count=5, width=640, height=360, points=pts)
        path = compute_crop_path(track, crop_w=1280, crop_h=720, dead_zone=0, end_zone_frac=0)

        # Degenerate case: crop is larger than source.
        # Centres are clamped within [0, src_w] and [0, src_h].
        assert np.all(path.cxs >= 0)
        assert np.all(path.cxs <= 640)
        assert np.all(path.cys >= 0)
        assert np.all(path.cys <= 360)


# ---------------------------------------------------------------------------
# CropPath dataclass
# ---------------------------------------------------------------------------


class TestCropPathDataclass:
    """Basic CropPath structure tests."""

    def test_output_shape(self):
        track = _make_track(frame_count=50)
        path = compute_crop_path(track)
        assert path.cxs.shape == (50,)
        assert path.cys.shape == (50,)
        assert path.zooms.shape == (50,)
        assert path.base_w == 1280
        assert path.base_h == 720

    def test_dtype_is_float64(self):
        track = _make_track(frame_count=10)
        path = compute_crop_path(track)
        assert path.cxs.dtype == np.float64
        assert path.cys.dtype == np.float64
        assert path.zooms.dtype == np.float64

    def test_backward_compat_aliases(self):
        track = _make_track(frame_count=10)
        path = compute_crop_path(track)
        assert path.crop_w == 1280
        assert path.crop_h == 720


# ---------------------------------------------------------------------------
# _fill_gaps_hold_snap
# ---------------------------------------------------------------------------


class TestFillGapsHoldSnap:
    """Adaptive gap fill: short gaps interpolate, long gaps hold briefly."""

    def test_short_gap_interpolates_directly(self):
        """Gap shorter than 2s: pure linear interpolation, no hold."""
        # 10-frame gap at 25fps = 0.4s → short gap
        raw = np.full((12, 2), np.nan)
        raw[0] = [100, 200]
        raw[11] = [200, 300]
        _fill_gaps_hold_snap(raw, fps=25.0, long_gap_s=2.0)

        # Frame 5 should be roughly halfway between 100 and 200
        assert 130 < raw[5, 0] < 170
        # Frame 1 should already have moved from 100
        assert raw[1, 0] > 100

    def test_long_gap_holds_then_interpolates(self):
        """Gap ≥ 2s: hold 30% then interpolate remaining 70%."""
        # 100-frame gap at 25fps = 4s → long gap
        raw = np.full((102, 2), np.nan)
        raw[0] = [100, 200]
        raw[101] = [200, 300]
        _fill_gaps_hold_snap(raw, fps=25.0, long_gap_s=2.0, long_hold_frac=0.3)

        # hold_end = 0 + int(101*0.3) = 30
        # Frames 1-29 should hold at 100
        np.testing.assert_array_equal(raw[1:30, 0], 100)
        # Frame 80 should be interpolating toward 200
        assert raw[80, 0] > 150

    def test_no_gaps(self):
        """Fully populated array is unchanged."""
        raw = np.column_stack([np.arange(10, dtype=float), np.arange(10, dtype=float)])
        original = raw.copy()
        _fill_gaps_hold_snap(raw)
        np.testing.assert_array_equal(raw, original)

    def test_extrapolates_edges(self):
        """Frames before first and after last valid point are held."""
        raw = np.full((10, 2), np.nan)
        raw[3] = [100, 200]
        raw[7] = [300, 400]
        _fill_gaps_hold_snap(raw)

        # Before first valid: hold
        np.testing.assert_array_equal(raw[0, 0], 100)
        np.testing.assert_array_equal(raw[2, 0], 100)
        # After last valid: hold
        np.testing.assert_array_equal(raw[8, 0], 300)
        np.testing.assert_array_equal(raw[9, 0], 300)


# ---------------------------------------------------------------------------
# _remove_aberrations
# ---------------------------------------------------------------------------


class TestRemoveAberrations:
    """Aberrant tracking burst removal."""

    def test_short_spike_removed(self):
        """A 3-frame burst far from the trajectory is removed."""
        raw = np.full((100, 2), np.nan)
        # Long stable segment at x=1000
        raw[0:40] = [1000, 2000]
        # 3-frame spike at x=3500 (2500px away)
        raw[40:43] = [3500, 2000]
        # Long stable segment continues at x=1000
        raw[43:100] = [1000, 2000]

        _remove_aberrations(raw, max_jump_px=500, max_burst_frames=10)

        # Spike should be gone
        assert np.all(np.isnan(raw[40:43, 0]))
        # Surrounding data untouched
        assert raw[39, 0] == 1000
        assert raw[43, 0] == 1000

    def test_long_segment_kept(self):
        """A segment longer than max_burst_frames is never removed."""
        raw = np.full((100, 2), np.nan)
        raw[0:30] = [1000, 2000]
        # 20-frame segment at distant position (> max_burst_frames=10)
        raw[30:50] = [3500, 2000]
        raw[50:100] = [1000, 2000]

        _remove_aberrations(raw, max_jump_px=500, max_burst_frames=10)

        # Long segment is kept even though it's far away
        assert np.isfinite(raw[35, 0])
        assert raw[35, 0] == 3500

    def test_nearby_short_segment_kept(self):
        """A short segment close to neighbours is kept."""
        raw = np.full((100, 2), np.nan)
        raw[0:40] = [1000, 2000]
        # 5-frame segment nearby (only 200px away, < max_jump_px)
        raw[40:45] = [1200, 2000]
        raw[45:100] = [1000, 2000]

        _remove_aberrations(raw, max_jump_px=500, max_burst_frames=10)

        # Close segment is kept
        assert np.isfinite(raw[42, 0])
        assert raw[42, 0] == 1200

    def test_multiple_spikes(self):
        """Multiple aberrant bursts are all removed."""
        raw = np.full((200, 2), np.nan)
        raw[0:50] = [1000, 2000]
        raw[50:53] = [4000, 2000]  # spike 1
        raw[53:100] = [1000, 2000]
        raw[100:102] = [3000, 500]  # spike 2
        raw[102:200] = [1000, 2000]

        _remove_aberrations(raw, max_jump_px=500, max_burst_frames=10)

        assert np.all(np.isnan(raw[50:53, 0]))
        assert np.all(np.isnan(raw[100:102, 0]))
        assert raw[49, 0] == 1000
        assert raw[53, 0] == 1000

    def test_no_valid_data_is_noop(self):
        """All-NaN input does not crash."""
        raw = np.full((10, 2), np.nan)
        _remove_aberrations(raw)  # should not raise


# ---------------------------------------------------------------------------
# _compute_zoom
# ---------------------------------------------------------------------------


class TestComputeZoom:
    """Dynamic zoom based on ball tracking status."""

    def test_all_tracked_stays_at_one(self):
        """All frames have ball → zoom stays 1.0 throughout."""
        has_ball = np.ones(100, dtype=bool)
        zooms = _compute_zoom(has_ball, 100, fps=25.0, max_zoom=2.5)
        np.testing.assert_array_almost_equal(zooms, 1.0)

    def test_short_loss_no_zoom(self):
        """Ball lost for less than delay → zoom stays near 1.0."""
        has_ball = np.ones(100, dtype=bool)
        # Lose ball for 5 frames (0.2s < 0.5s delay)
        has_ball[50:55] = False
        zooms = _compute_zoom(has_ball, 100, fps=25.0, delay_s=0.5, max_zoom=2.5)
        # Should barely deviate from 1.0
        assert np.all(zooms < 1.05)

    def test_long_loss_zooms_out(self):
        """Ball lost for > delay → zoom ramps toward max_zoom."""
        has_ball = np.ones(100, dtype=bool)
        # Lose ball from frame 10 to 99 (3.6s >> 0.5s delay)
        has_ball[10:] = False
        zooms = _compute_zoom(
            has_ball, 100, fps=25.0, delay_s=0.5, max_zoom=2.5,
            zoom_out_alpha=0.06,
        )

        # First 10 frames: 1.0
        np.testing.assert_array_almost_equal(zooms[:10], 1.0)
        # By frame 99, should have zoomed out significantly
        assert zooms[99] > 1.5

    def test_reacquire_zooms_back_in(self):
        """Ball re-acquired after loss → zoom eases back to 1.0."""
        has_ball = np.ones(200, dtype=bool)
        has_ball[10:60] = False  # 2s loss
        zooms = _compute_zoom(
            has_ball, 200, fps=25.0, delay_s=0.5, max_zoom=2.5,
            zoom_out_alpha=0.06, zoom_in_alpha=0.04,
        )

        # Peak zoom somewhere in the loss region
        peak = zooms[60]
        assert peak > 1.2
        # After re-acquisition, zoom should decrease toward 1.0
        assert zooms[199] < peak
        assert zooms[199] < 1.1  # close to 1.0 after 140 frames

    def test_max_zoom_respected(self):
        """Zoom never exceeds max_zoom."""
        has_ball = np.zeros(1000, dtype=bool)
        zooms = _compute_zoom(
            has_ball, 1000, fps=25.0, delay_s=0.5, max_zoom=2.0,
            zoom_out_alpha=0.1,
        )
        assert np.all(zooms <= 2.01)  # small tolerance


# ---------------------------------------------------------------------------
# compute_crop_path — zone-based stabilisation
# ---------------------------------------------------------------------------


class TestCropPathZones:
    """Zone-aware camera: locked at ends, fast in transit."""

    def test_left_end_zone_locks_camera(self):
        """Ball deep in left end zone → camera locks near left edge."""
        # Ball at x=800, well inside left end zone (end_left=1152)
        pts = _make_points_at(800, 2304, 0, 50)
        track = _make_track(frame_count=50, width=4608, height=4608, points=pts)
        path_zones = compute_crop_path(
            track, crop_w=1280, crop_h=720, dead_zone=0, end_zone_frac=0.25,
        )
        path_no_zones = compute_crop_path(
            track, crop_w=1280, crop_h=720, dead_zone=0, end_zone_frac=0,
        )
        # With zones, camera should be closer to the left edge
        assert path_zones.cxs[25] <= path_no_zones.cxs[25]

    def test_right_end_zone_locks_camera(self):
        """Ball deep in right end zone → camera locks near right edge."""
        # Ball at x=3800, inside right end zone (end_right=3456)
        pts = _make_points_at(3800, 2304, 0, 50)
        track = _make_track(frame_count=50, width=4608, height=4608, points=pts)
        path_zones = compute_crop_path(
            track, crop_w=1280, crop_h=720, dead_zone=0, end_zone_frac=0.25,
        )
        path_no_zones = compute_crop_path(
            track, crop_w=1280, crop_h=720, dead_zone=0, end_zone_frac=0,
        )
        # With zones, camera should be closer to the right edge
        assert path_zones.cxs[25] >= path_no_zones.cxs[25]

    def test_end_zone_keeps_ball_visible(self):
        """Ball moves within end zone → camera nudges to keep it in frame."""
        # Ball starts at x=200 (near left wall) then shifts to x=1100
        pts = _make_points_at(200, 2304, 0, 25)
        pts += _make_points_at(1100, 2304, 25, 50)
        track = _make_track(frame_count=50, width=4608, height=4608, points=pts)
        path = compute_crop_path(
            track, crop_w=1280, crop_h=720, dead_zone=0, end_zone_frac=0.25,
        )
        # Ball at 1100 must be inside crop window:
        # crop left = cxs - 640, crop right = cxs + 640
        assert path.cxs[49] - 640 <= 1100
        assert path.cxs[49] + 640 >= 1100

    def test_middle_zone_convergence(self):
        """Ball at fixed position in middle → camera converges there."""
        pts = _make_points_at(2304, 2304, 0, 200)
        track = _make_track(frame_count=200, width=4608, height=4608, points=pts)
        path_zones = compute_crop_path(
            track, crop_w=1280, crop_h=720, dead_zone=0, end_zone_frac=0.25,
        )
        path_no_zones = compute_crop_path(
            track, crop_w=1280, crop_h=720, dead_zone=0, end_zone_frac=0,
        )
        # Both should converge to the same centre position (2304)
        assert abs(path_zones.cxs[199] - path_no_zones.cxs[199]) < 5

    def test_zone_boundary_is_continuous(self):
        """Camera position is continuous across zone boundary."""
        # Ball moves from end zone into middle zone
        pts = []
        for i in range(100):
            x = 500 + i * 20  # 500 → 2480, crosses end_left=1152
            pts.append(TrackPoint(frame=i, x=float(x), y=2304.0,
                                  confidence=0.9, state="tracking"))
        track = _make_track(frame_count=100, width=4608, height=4608, points=pts)
        path = compute_crop_path(
            track, crop_w=1280, crop_h=720, dead_zone=0,
            max_vel=9999, end_zone_frac=0.25,
        )
        # No large jumps between consecutive frames
        diffs = np.abs(np.diff(path.cxs))
        assert np.all(diffs < 80)  # smooth transitions

    def test_fast_break_pans_quickly(self):
        """Ball goes from one end to the other → camera follows fast."""
        # Ball at right end for 50 frames, then at left end
        pts = _make_points_at(4000, 2304, 0, 50)
        pts += _make_points_at(500, 2304, 50, 100)
        track = _make_track(frame_count=100, width=4608, height=4608, points=pts)
        path = compute_crop_path(
            track, crop_w=1280, crop_h=720, dead_zone=0, end_zone_frac=0.25,
        )
        # Camera should be near right end at frame 49 (centre > 3000)
        assert path.cxs[49] > 3000
        # Camera should have panned significantly toward left by frame 75
        assert path.cxs[75] < path.cxs[49]

    def test_out_of_bounds_points_filtered(self):
        """Points with x < 0 are discarded, not used for crop path."""
        pts = _make_points_at(2304, 2304, 0, 50)
        # Add bad points with negative x (Kalman drift)
        pts += [TrackPoint(frame=i, x=-5000.0, y=2304.0,
                           confidence=0.5, state="gap_bridge")
                for i in range(50, 100)]
        track = _make_track(frame_count=100, width=4608, height=4608, points=pts)
        path = compute_crop_path(
            track, crop_w=1280, crop_h=720, dead_zone=0, end_zone_frac=0,
        )
        # All crop positions must keep crop within frame
        for i in range(len(path.cxs)):
            z = path.zooms[i]
            hw = path.base_w * z / 2
            assert path.cxs[i] >= hw - 1  # small tolerance
            assert path.cxs[i] + hw <= 4608 + 1

    def test_gap_bridge_points_ignored(self):
        """Only 'tracking' state points are used for the crop path."""
        # tracking points at centre
        pts = _make_points_at(2304, 2304, 0, 50)
        # gap_bridge points far away (should be ignored)
        pts += [TrackPoint(frame=i, x=100.0, y=2304.0,
                           confidence=0.5, state="gap_bridge")
                for i in range(50, 100)]
        track = _make_track(frame_count=100, width=4608, height=4608, points=pts)
        path = compute_crop_path(
            track, crop_w=1280, crop_h=720, dead_zone=0, end_zone_frac=0,
        )
        # After frame 50, gap_bridge points should be ignored.
        # Camera should stay near centre (2304), not jump to x=100.
        assert path.cxs[75] > 1640


# ---------------------------------------------------------------------------
# FFmpegWriter
# ---------------------------------------------------------------------------


class TestFFmpegWriter:
    """FFmpegWriter unit tests (mock subprocess)."""

    @patch("wpv.render.reframe.shutil.which", return_value=None)
    def test_raises_without_ffmpeg(self, mock_which):
        """Constructor raises if ffmpeg is not found."""
        with pytest.raises(RuntimeError, match="ffmpeg not found"):
            FFmpegWriter("/tmp/out.mp4", 1280, 720, 25.0)

    @patch("wpv.render.reframe.tempfile.TemporaryFile")
    @patch("wpv.render.reframe.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("wpv.render.reframe.subprocess.Popen")
    def test_context_manager_starts_and_stops(self, MockPopen, mock_which, MockTmpFile):
        """Writer starts ffmpeg on enter, closes stdin on exit."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        MockPopen.return_value = mock_proc
        MockTmpFile.return_value = MagicMock()

        writer = FFmpegWriter("/tmp/out.mp4", 1280, 720, 25.0)
        with writer:
            pass

        mock_proc.stdin.close.assert_called_once()
        mock_proc.wait.assert_called_once()

    @patch("wpv.render.reframe.tempfile.TemporaryFile")
    @patch("wpv.render.reframe.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("wpv.render.reframe.subprocess.Popen")
    def test_write_frame_sends_bytes(self, MockPopen, mock_which, MockTmpFile):
        """write_frame sends raw bytes to stdin pipe."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        MockPopen.return_value = mock_proc
        MockTmpFile.return_value = MagicMock()

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        with FFmpegWriter("/tmp/out.mp4", 1280, 720, 25.0) as w:
            w.write_frame(frame)

        mock_proc.stdin.write.assert_called_once()
        written = mock_proc.stdin.write.call_args[0][0]
        assert len(written) == 1280 * 720 * 3

    @patch("wpv.render.reframe.tempfile.TemporaryFile")
    @patch("wpv.render.reframe.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("wpv.render.reframe.subprocess.Popen")
    def test_audio_source_adds_map_flags(self, MockPopen, mock_which, MockTmpFile):
        """When source_path is given, ffmpeg command includes -map and -c:a."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        MockPopen.return_value = mock_proc
        MockTmpFile.return_value = MagicMock()

        with FFmpegWriter("/tmp/out.mp4", 1280, 720, 25.0, source_path="/src.mp4"):
            pass

        cmd = MockPopen.call_args[0][0]
        assert "-map" in cmd
        assert "1:a?" in cmd
        assert "-c:a" in cmd

    @patch("wpv.render.reframe.tempfile.TemporaryFile")
    @patch("wpv.render.reframe.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("wpv.render.reframe.subprocess.Popen")
    def test_ffmpeg_error_raises(self, MockPopen, mock_which, MockTmpFile):
        """Non-zero exit code raises RuntimeError."""
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        MockPopen.return_value = mock_proc

        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b"encode error"
        MockTmpFile.return_value = mock_stderr

        with pytest.raises(RuntimeError, match="ffmpeg exited with code 1"):
            with FFmpegWriter("/tmp/out.mp4", 1280, 720, 25.0):
                pass


# ---------------------------------------------------------------------------
# CropRenderer
# ---------------------------------------------------------------------------


class TestCropRenderer:
    """CropRenderer integration tests with mocked I/O."""

    def _make_mock_reader(self, width=4608, height=4608, fps=25.0, frame_count=10):
        reader = MagicMock()
        reader.fps = fps
        reader.frame_count = frame_count
        reader.width = width
        reader.height = height

        def _seq_frames(start=0, end=None):
            for i in range(frame_count):
                yield i, np.zeros((height, width, 3), dtype=np.uint8)

        reader.sequential_frames.side_effect = _seq_frames
        return reader

    @patch("wpv.render.reframe.detect_pool_bounds")
    @patch("wpv.render.reframe.FFmpegWriter")
    @patch("wpv.tracking.video_reader.VideoReader")
    def test_render_writes_all_frames(self, MockReader, MockWriter, mock_detect):
        """Renderer iterates all frames and writes each one."""
        pool_bounds = PoolBounds(100, 4500, 2100, 3200)
        mock_detect.return_value = pool_bounds
        mock_reader = self._make_mock_reader(frame_count=5)
        MockReader.return_value = mock_reader

        mock_writer_instance = MagicMock()
        mock_writer_instance.__enter__ = MagicMock(return_value=mock_writer_instance)
        mock_writer_instance.__exit__ = MagicMock(return_value=False)
        MockWriter.return_value = mock_writer_instance

        track = _make_track(
            frame_count=5, width=4608, height=4608,
            points=_make_points_at(2304, 2304, 0, 5),
        )

        # Compute crop path explicitly to pass to _run_python
        crop_path = compute_crop_path(track, pool_bounds=pool_bounds)

        renderer = CropRenderer(
            video_path="/fake/video.mp4",
            track=track,
            output_path="/tmp/out.mp4",
            pool_bounds=pool_bounds,
        )
        renderer._run_python(crop_path)

        assert mock_writer_instance.write_frame.call_count == 5

    @patch("wpv.render.reframe.detect_pool_bounds")
    @patch("wpv.render.reframe.FFmpegWriter")
    @patch("wpv.tracking.video_reader.VideoReader")
    def test_preview_mode_uses_fast_settings(self, MockReader, MockWriter, mock_detect):
        """Preview mode sets crf=28, preset=ultrafast, half-res."""
        mock_detect.return_value = PoolBounds(100, 4500, 2100, 3200)
        mock_reader = self._make_mock_reader(frame_count=2)
        MockReader.return_value = mock_reader

        mock_writer_instance = MagicMock()
        mock_writer_instance.__enter__ = MagicMock(return_value=mock_writer_instance)
        mock_writer_instance.__exit__ = MagicMock(return_value=False)
        MockWriter.return_value = mock_writer_instance

        track = _make_track(
            frame_count=2, width=4608, height=4608,
            points=_make_points_at(2304, 2304, 0, 2),
        )

        renderer = CropRenderer(
            video_path="/fake/video.mp4",
            track=track,
            output_path="/tmp/out.mp4",
            preview=True,
        )
        renderer.run()

        # Check FFmpegWriter was called with preview settings
        call_kwargs = MockWriter.call_args[1]
        assert call_kwargs["crf"] == 28
        assert call_kwargs["preset"] == "ultrafast"
        assert call_kwargs["width"] == 640
        assert call_kwargs["height"] == 360


# ---------------------------------------------------------------------------
# PoolBounds — pool-mask-aware camera
# ---------------------------------------------------------------------------


class TestPoolBounds:
    """Pool-bounds-aware crop path computation."""

    # Realistic pool bounds for a 4608×4608 fisheye frame:
    # Pool water: x≈100..4500, y≈2100..3200 (after goal padding on x)
    POOL = PoolBounds(x_min=100, x_max=4500, y_min=2100, y_max=3200)

    def test_nan_fallback_uses_pool_centre(self):
        """No tracking points → crop centres on pool centre, not frame centre."""
        track = _make_track(frame_count=10, width=4608, height=4608)
        path = compute_crop_path(
            track, crop_w=1280, crop_h=720, dead_zone=0, end_zone_frac=0,
            pool_bounds=self.POOL,
        )
        # Pool centre x = (100+4500)/2 = 2300, y = (2100+3200)/2 = 2650
        assert path.cxs[0] == pytest.approx(2300, abs=1)
        assert path.cys[0] == pytest.approx(2650, abs=1)

    def test_y_clamp_stays_in_pool_tall(self):
        """When pool is taller than crop, y stays within pool bounds."""
        # Pool taller than crop (1100 > 720 in self.POOL)
        pts = _make_points_at(2300, 1000, 0, 50)
        track = _make_track(frame_count=50, width=4608, height=4608, points=pts)
        path = compute_crop_path(
            track, crop_w=1280, crop_h=720, dead_zone=0, end_zone_frac=0,
            pool_bounds=self.POOL,
        )
        # Centre must be within [y_min + crop_h/2, y_max - crop_h/2]
        assert np.all(path.cys >= self.POOL.y_min + 720 / 2)
        assert np.all(path.cys <= self.POOL.y_max - 720 / 2)

    def test_y_clamp_centres_on_short_pool(self):
        """When pool is shorter than crop, crop centres on pool vertically."""
        # Realistic pool: only 413px tall (< 720px crop)
        short_pool = PoolBounds(x_min=604, x_max=3876, y_min=2323, y_max=2736)
        pts = _make_points_at(2000, 2500, 0, 50)
        track = _make_track(frame_count=50, width=4608, height=4608, points=pts)
        path = compute_crop_path(
            track, crop_w=1280, crop_h=720, dead_zone=0, end_zone_frac=0,
            pool_bounds=short_pool,
        )
        # Pool centre = (2323+2736)/2 = 2529.5 → all cys pinned there
        np.testing.assert_array_almost_equal(path.cys, short_pool.cy, decimal=0)

    def test_end_zone_uses_pool_geometry(self):
        """End zone boundaries are relative to pool, not frame."""
        # Ball deep in left pool area (x=300, just inside pool)
        pts = _make_points_at(300, 2650, 0, 50)
        track = _make_track(frame_count=50, width=4608, height=4608, points=pts)
        path = compute_crop_path(
            track, crop_w=1280, crop_h=720, dead_zone=0, end_zone_frac=0.25,
            pool_bounds=self.POOL,
        )
        # Camera should lock near pool left edge:
        # cam_left = pool.x_min + crop_w/2 = 100 + 640 = 740
        # Centre should be near 740, definitely < 1040
        assert path.cxs[49] < 1040

    def test_end_zone_y_targets_goal(self):
        """In end zones the camera vertical target is pool centre (goal y)."""
        # Ball in left end zone but at odd y position
        pts = _make_points_at(300, 2200, 0, 200)
        track = _make_track(frame_count=200, width=4608, height=4608, points=pts)
        path = compute_crop_path(
            track, crop_w=1280, crop_h=720, dead_zone=0, end_zone_frac=0.25,
            pool_bounds=self.POOL,
        )
        # In end zone, y target is pool.cy = 2650
        assert abs(path.cys[199] - 2650) < 100

    def test_transit_zone_follows_ball_y(self):
        """In the transit zone the camera tracks ball y, not goal centre."""
        # Ball in middle of pool at non-centre y
        pts = _make_points_at(2300, 2900, 0, 200)
        track = _make_track(frame_count=200, width=4608, height=4608, points=pts)
        path = compute_crop_path(
            track, crop_w=1280, crop_h=720, dead_zone=0, end_zone_frac=0.25,
            pool_bounds=self.POOL,
        )
        # In transit, y should converge toward 2900 (clamped ≤ 2840), not pool.cy=2650
        assert path.cys[199] > 2650 + 50

    def test_pool_bounds_properties(self):
        """PoolBounds properties compute correctly."""
        pb = PoolBounds(100, 4500, 2100, 3200)
        assert pb.width == 4400
        assert pb.height == 1100
        assert pb.cx == 2300.0
        assert pb.cy == 2650.0

    def test_from_polygon(self):
        """from_polygon converts a 4-point polygon to PoolBounds with padding."""
        points = [[754, 2736], [1903, 2373], [2526, 2377], [3726, 2672]]
        pb = PoolBounds.from_polygon(points, goal_pad_px=150, camera_pad_px=50)
        # x: min=754 - 150 = 604, max=3726 + 150 = 3876
        assert pb.x_min == 604
        assert pb.x_max == 3876
        # y: min=2373 - 50 = 2323 (camera pad), max=2736 (no pad)
        assert pb.y_min == 2323
        assert pb.y_max == 2736

    def test_from_polygon_clamps_to_frame(self):
        """from_polygon doesn't go negative or past frame edge."""
        points = [[50, 100], [4590, 4590], [2000, 50], [2500, 4500]]
        pb = PoolBounds.from_polygon(
            points, goal_pad_px=200, camera_pad_px=100, frame_w=4608, frame_h=4608,
        )
        assert pb.x_min == 0       # clamped: 50 - 200 = -150 → 0
        assert pb.x_max == 4608    # clamped: 4590 + 200 = 4790 → 4608
        assert pb.y_min == 0       # clamped: 50 - 100 = -50 → 0

    def test_load_game_mask_bounds(self, tmp_path):
        """load_game_mask_bounds reads JSON and produces correct bounds."""
        import json
        masks = {"clip_000": [[754, 2736], [1903, 2373], [2526, 2377], [3726, 2672]]}
        mask_file = tmp_path / "game_masks.json"
        mask_file.write_text(json.dumps(masks))

        pb = load_game_mask_bounds(mask_file, "clip_000")
        assert pb.x_min == 604
        assert pb.x_max == 3876
        assert pb.y_min == 2323
        assert pb.y_max == 2736


# ---------------------------------------------------------------------------
# Fisheye remap
# ---------------------------------------------------------------------------


class TestFisheyeRemap:
    """Fisheye-to-perspective remapping math."""

    INTRINSICS = FisheyeIntrinsics(
        focal_length_px=1466.0,
        center_x=2304.0,
        center_y=2304.0,
    )

    def test_center_pixel_maps_to_look_direction(self):
        """Output centre maps back to the fisheye look-direction pixel."""
        cx_fish, cy_fish = 2500.0, 2600.0
        out_w, out_h = 640, 360
        map_x, map_y = build_fisheye_to_perspective_maps(
            cx_fish, cy_fish, out_w, out_h, 70.0, self.INTRINSICS,
        )
        # Centre of output = (320, 180)
        centre_src_x = map_x[out_h // 2, out_w // 2]
        centre_src_y = map_y[out_h // 2, out_w // 2]
        assert abs(centre_src_x - cx_fish) < 1.0
        assert abs(centre_src_y - cy_fish) < 1.0

    def test_map_shape_and_dtype(self):
        """Remap arrays have correct shape and float32 dtype."""
        map_x, map_y = build_fisheye_to_perspective_maps(
            2304.0, 2304.0, 1280, 720, 70.0, self.INTRINSICS,
        )
        assert map_x.shape == (720, 1280)
        assert map_y.shape == (720, 1280)
        assert map_x.dtype == np.float32
        assert map_y.dtype == np.float32

    def test_hfov_widens_coverage(self):
        """Wider FOV → remap touches a larger area of the source frame."""
        map_x_narrow, _ = build_fisheye_to_perspective_maps(
            2304.0, 2304.0, 640, 360, 50.0, self.INTRINSICS,
        )
        map_x_wide, _ = build_fisheye_to_perspective_maps(
            2304.0, 2304.0, 640, 360, 110.0, self.INTRINSICS,
        )
        range_narrow = map_x_narrow.max() - map_x_narrow.min()
        range_wide = map_x_wide.max() - map_x_wide.min()
        assert range_wide > range_narrow

    def test_zoom_scales_hfov(self):
        """fisheye_hfov_for_zoom multiplies base by zoom."""
        assert fisheye_hfov_for_zoom(70.0, 2.0) == 140.0
        assert fisheye_hfov_for_zoom(70.0, 1.0) == 70.0
        assert fisheye_hfov_for_zoom(70.0, 0.5) == 35.0

    def test_optical_axis_edge_case(self):
        """Looking straight along optical axis (cx=ocx, cy=ocy) doesn't crash."""
        map_x, map_y = build_fisheye_to_perspective_maps(
            2304.0, 2304.0, 640, 360, 70.0, self.INTRINSICS,
        )
        # Centre should map to (2304, 2304)
        assert abs(map_x[180, 320] - 2304.0) < 1.0
        assert abs(map_y[180, 320] - 2304.0) < 1.0
        # No NaNs
        assert not np.any(np.isnan(map_x))
        assert not np.any(np.isnan(map_y))

    @patch("wpv.render.reframe.detect_pool_bounds")
    @patch("wpv.render.reframe.FFmpegWriter")
    @patch("wpv.tracking.video_reader.VideoReader")
    def test_undistort_false_uses_legacy(self, MockReader, MockWriter, mock_detect):
        """When undistort=False, cv2.remap is NOT called (legacy crop path)."""
        import cv2

        pool_bounds = PoolBounds(100, 4500, 2100, 3200)
        mock_detect.return_value = pool_bounds

        mock_reader = MagicMock()
        mock_reader.fps = 25.0
        frame = np.zeros((4608, 4608, 3), dtype=np.uint8)

        def _seq_frames(start=0, end=None):
            for i in range(3):
                yield i, frame

        mock_reader.sequential_frames.side_effect = _seq_frames
        MockReader.return_value = mock_reader

        mock_writer_instance = MagicMock()
        mock_writer_instance.__enter__ = MagicMock(return_value=mock_writer_instance)
        mock_writer_instance.__exit__ = MagicMock(return_value=False)
        MockWriter.return_value = mock_writer_instance

        track = _make_track(
            frame_count=3, width=4608, height=4608,
            points=_make_points_at(2304, 2304, 0, 3),
        )

        renderer = CropRenderer(
            video_path="/fake/video.mp4",
            track=track,
            output_path="/tmp/out.mp4",
            pool_bounds=pool_bounds,
            undistort=False,
            preview=True,  # force Python path so we can test crop vs remap
        )

        with patch.object(cv2, "remap", wraps=cv2.remap) as mock_remap:
            renderer.run()
            mock_remap.assert_not_called()
        # Should have written frames via legacy crop+resize
        assert mock_writer_instance.write_frame.call_count == 3
