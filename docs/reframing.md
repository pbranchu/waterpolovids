# Reframing: Crop-and-Pan Camera from Fisheye to 16:9

## Overview

The reframing stage takes a stationary 4608x4608 fisheye video (Insta360 X5 single-lens, ~180° FOV) and produces a 1280x720 cropped-and-panned video that follows the ball, simulating a human camera operator.

The source video is **not** dewarped. All coordinates (ball tracking, pool boundaries, crop positions) are in the native fisheye pixel space. Because the crop window is small relative to the full frame (~28% width), distortion within a single crop is moderate and acceptable.

## Inputs

| Input | Source | Description |
|-------|--------|-------------|
| Source video | `PRO_VID_*.mp4` | 4608x4608 fisheye, 25fps |
| Track data | `work/<clip>.json` | Ball positions per frame from the tracker |
| Pool boundaries | `game_masks.json` | Manually-set 4-point polygon per clip |

### Pool Boundaries

The pool appears as a trapezoid in the fisheye frame due to perspective (wider at the ends, narrower near the camera). A 4-point polygon is manually drawn through the labeling UI (`scripts/label_ball.py`) and saved in `data/labeling/game_masks.json`.

The polygon is converted to a `PoolBounds` bounding box with two types of padding:

- **Goal padding** (x-axis, ±150px): the water mask ends at the goal line, but the goals themselves extend beyond. The padding ensures the crop can show the full goal area.
- **Camera padding** (y-axis, -50px toward camera): accounts for residual fisheye distortion at the near edge of the pool.

When the pool is shorter than the crop height (typical: pool is ~413px tall, crop is 720px), the crop is vertically centred on the pool and pinned — it never pans vertically, always showing the full pool with equal margins above and below.

If no manual mask is available, `detect_pool_bounds()` auto-detects the pool using blue-water thresholding from the first frame (fallback).

## Pipeline: Three Passes

### Pass 1: Fill Raw Ball Positions

**Goal:** Build a clean per-frame ball position array from noisy, sparse tracker output.

#### 1a. Filter to high-confidence points only

Only "tracking" state points are used. "gap_bridge" (Kalman predictions during tracking loss) and "action_proxy" points are discarded — they tend to drift, especially over long gaps where the Kalman filter extrapolates wildly.

Points with coordinates outside the frame boundaries (x < 0 or x > frame_width) are also discarded.

#### 1b. Remove aberrant tracking bursts

`_remove_aberrations(max_jump_px=500, max_burst_frames=10)`

The tracker occasionally locks onto false positives (yellow caps, lane markers, spectators) for a few frames before returning to the real ball. These appear as short bursts of positions that teleport far from the surrounding trajectory.

Algorithm:
1. Segment valid points into contiguous runs, splitting at any jump > 500px between consecutive frames.
2. For each short run (≤ 10 frames), find the nearest long run before and after.
3. If the short run's mean position is > 500px from all neighbouring long runs, remove it (set to NaN).
4. If no long runs exist at all, keep everything.

#### 1c. Adaptive gap fill

`_fill_gaps_hold_snap(fps, long_gap_s=2.0, long_hold_frac=0.3)`

After filtering and aberration removal, many frames have no ball position. Gaps are filled differently based on duration:

**Short gaps (< 2 seconds):** Pure linear interpolation between the last known and next known position. The ball was briefly lost (e.g., obscured by a player, underwater, between frames) — the camera should move smoothly toward the next known position immediately.

**Long gaps (≥ 2 seconds):** 30% hold at the last known position, then 70% linear interpolation to the next. This handles scenarios like a goalkeeper holding the ball (brief hold is natural) before throwing it to the other end (camera then pans).

**Edge extrapolation:** Frames before the first known point hold at the first known position. Frames after the last known point hold at the last known position.

#### 1d. NaN fallback

Any remaining unknown frames (no tracking data and no neighbours to interpolate from) default to the pool centre, not the frame centre.

### Pass 2: Zone-Aware Smoothing

**Goal:** Simulate broadcast-quality camera movement — steady at the goals, responsive in transitions.

The pool is divided into three zones based on the pool bounds:

```
|  Left End Zone (25%)  |   Transit Zone (50%)   |  Right End Zone (25%)  |
|  camera: locked       |  camera: tracks ball   |  camera: locked        |
|  alpha: 0.02          |  alpha: 0.15           |  alpha: 0.02           |
|  dead zone: active    |  dead zone: none       |  dead zone: active     |
|  y-target: goal_y     |  y-target: ball_y      |  y-target: goal_y      |
```

A 200px transition band around each zone boundary blends all parameters (target position, alpha, max velocity, dead zone) linearly, ensuring no discontinuity.

#### End Zone Behaviour

When the ball is in an end zone (outer 25% of the pool on each side):

- **X target:** Camera locks to a fixed position showing the goal edge (`pool_x_min + crop_w/2` for left, `pool_x_max - crop_w/2` for right). If the ball would be outside the visible crop, the camera nudges the minimum amount needed to keep the ball within `margin` pixels (100px) of the crop edge. It does NOT centre on the ball.
- **Y target:** Pool vertical centre (goal y-position). In end zones the camera holds a stable vertical frame showing the goal.
- **Smoothing:** Very sluggish (alpha=0.02). Large dead zone. The camera barely moves unless it must.

This produces the broadcast look: a wide, stable shot of the goal area where the action unfolds.

#### Transit Zone Behaviour

When the ball is in the middle 50% of the pool:

- **X target:** Ball x-position directly (camera centres on ball).
- **Y target:** Ball y-position directly.
- **Smoothing:** Responsive (alpha=0.15). No dead zone. High max velocity (≥200 px/frame). The camera follows the ball aggressively.

This handles fast breaks, counter-attacks, and cross-pool transitions. The camera pans quickly to keep up.

#### Transition Band

The 200px band around each zone boundary interpolates:
- `target = locked * (1-t) + ball * t` where t goes 0→1 from end zone to transit
- `alpha = EZ_ALPHA * (1-t) + TR_ALPHA * t`
- `max_vel = base * (1-t) + fast * t`
- `dead_zone = dz * (1-t) + 0 * t`

This prevents any visible jerk when the ball crosses a zone boundary.

### Pass 3: Clamp

Convert smoothed centre positions to top-left crop origins, then clamp:

- **X:** Within `[0, src_w - crop_w]` (full source frame)
- **Y:** When pool height ≥ crop height: within `[pool.y_min, pool.y_max - crop_h]`. When pool height < crop height (typical): pinned at `pool.cy - crop_h/2` (crop centred on pool).

## Rendering

### Fast Path: Native ffmpeg Crop Filter

The crop positions are encoded as an ffmpeg expression using a balanced binary tree of `if(lt(n,...), ...)` comparisons. Keyframes are sampled every 25 frames with linear interpolation between them.

This keeps the expression nesting depth at O(log N), well within ffmpeg's limits, and lets ffmpeg handle decode → crop → encode entirely in C. This is 10-50x faster than reading frames in Python.

```
ffmpeg -y -i source.mp4 \
  -vf "crop=1280:720:<x_expr>:<y_expr>" \
  -c:v libx264 -crf 20 -preset medium \
  -pix_fmt yuv420p -c:a aac \
  -movflags +faststart output.mp4
```

### Slow Path: Python Frame-by-Frame

Used only for preview mode (half-res, crf 28, ultrafast preset). Reads each frame with OpenCV, crops with numpy slicing, writes via a pipe to ffmpeg.

## Key Design Decisions

### Why not dewarp first?

Dewarping the full 4608x4608 fisheye to rectilinear would require either:
- A full-frame remap (expensive, introduces interpolation artifacts)
- Multiple perspective projections stitched together

Since we're only extracting a 1280x720 window, the distortion within that window is acceptable. All coordinates (tracking, pool bounds, crop) are consistently in fisheye space.

### Why manual pool boundaries?

Auto-detection via blue-water thresholding works for most daylight games but fails on:
- Night games with artificial lighting
- Pools with non-standard water colour
- Frames where reflections dominate

Manual polygons from the labeling UI are reliable across all conditions and also serve as the detection mask for ball tracking.

### Why hold-then-interpolate instead of pure interpolation?

For long gaps (≥2s), a brief hold is natural — the ball is often stationary (goalkeeper, set play, timeout) before a long throw. Pure interpolation would start the camera moving before the ball does.

For short gaps (<2s), the ball is just briefly lost and likely still moving. Pure interpolation gives the smoothest result.

### Why not use future ball position more aggressively?

We already do — the gap fill interpolates toward the next known position. The zone smoothing then applies appropriate dynamics. A goalkeeper throw to the other end produces:
1. Brief 30% hold at the goalkeeper's end
2. Interpolation starts moving toward the receiver
3. As positions enter the transit zone, the camera ramps up to high-speed panning
4. Camera arrives at the receiver's end zone and locks

This naturally follows the action without explicit look-ahead logic.

## File Reference

| File | Purpose |
|------|---------|
| `src/wpv/render/reframe.py` | All crop path computation, smoothing, rendering |
| `src/wpv/tracking/detector.py` | `detect_pool_mask()` — auto-detection fallback |
| `data/labeling/game_masks.json` | Manual pool boundary polygons per clip |
| `tests/test_reframe.py` | 47 tests covering all passes and edge cases |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `crop_w` | 1280 | Output crop width |
| `crop_h` | 720 | Output crop height |
| `end_zone_frac` | 0.25 | Fraction of pool width for each end zone |
| `goal_pad_px` | 150 | X-axis extension beyond water mask for goals |
| `camera_pad_px` | 50 | Y-axis extension toward camera for distortion |
| `long_gap_s` | 2.0 | Gap duration threshold for hold behaviour |
| `long_hold_frac` | 0.3 | Fraction of long gap spent holding |
| `EZ_ALPHA` | 0.02 | End zone smoothing factor |
| `TR_ALPHA` | 0.15 | Transit zone smoothing factor |
| `TR_MAX_VEL` | 200+ | Transit max camera velocity (px/frame) |
| `margin` | 100 | Min distance from ball to crop edge in end zones |
| `band` | 200 | Transition band width between zones (px) |
| `max_jump_px` | 500 | Aberration detection: max allowed inter-frame jump |
| `max_burst_frames` | 10 | Aberration detection: max short segment length |
