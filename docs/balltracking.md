# Ball Detection & Tracking — Design Document

## Context

We're building an automated reframing pipeline for water polo footage shot with an Insta360 X5. The camera is stationary, mounted high and centered above the pool. Each video file covers one quarter (~15–20 minutes, ~15K–22K frames at 25fps). The frames are 4608×4608 equirectangular.

The ball (KAP7 HydroGrip, bright yellow with black seams) is typically 10–40 pixels in the equirectangular frame — small, often partially occluded by water, players, or reflections.

Ball detection is **not the end goal** — it feeds the virtual camera system that determines where to point the FOV for each frame. For this purpose, **high-confidence keyframes are more valuable than noisy per-frame detections**. The Kalman tracker and state machine (M3) will interpolate between confident detections and handle gaps.

## Current Setup

### Two-stage detection

1. **HSV candidate generation** (`detect_hsv_candidates` in `detector.py`)
   - Convert frame to HSV color space
   - Threshold with configurable bands (stored in `data/labeling/hsv_params.json`)
   - Morphological open (3×3) then close (5×5) to clean noise
   - Connected components analysis with area and circularity filters
   - Masked to pool region (auto-detected via blue thresholding) AND game area polygon (manually drawn once per clip)

2. **Reference-image scoring** (`BallReferenceScorer` in `detector.py`)
   - Uses a studio photo of the KAP7 ball (`data/ball_reference.webp`)
   - Builds a 2D Hue-Saturation histogram from the reference image
   - Scores each candidate via:
     - **Back-projection**: project the reference HS histogram onto the candidate region, take mean — measures what fraction of pixels have ball-like color
     - **Hue distance**: penalize candidates whose mean hue deviates from the ball's (H≈22)
   - Combined score rescaled to [0, 1]; threshold at ~0.6 separates balls from false positives

### Game area masking

The camera sees the entire pool deck, spectator areas, and beyond. False positives (yellow shirts, cones, equipment) are common outside the playing area. Each clip has a manually-drawn 4-point polygon defining the game area (goal line to goal line, pool width). Stored in `data/labeling/game_masks.json`.

Auto-detection of game area boundaries from yellow cones on the pool deck was implemented and works for ~70% of clips, with manual correction for the rest.

### NVDEC hardware decoding

The video files are H.265 encoded. Seeking in H.265 via OpenCV is extremely slow (~6s/frame) because it must decode from the nearest keyframe. We use NVIDIA's NVDEC hardware decoder via PyNvVideoCodec, which is ~54× faster for seek operations.

**Critical detail**: NVDEC outputs NV12 in BT.709 color space (standard for HD/4K H.265), while OpenCV's `COLOR_YUV2BGR_NV12` assumes BT.601. This caused a systematic color shift (HSV saturation off by ~32 on average) that broke detection. Fixed with a manual BT.709 NV12→BGR conversion.

### Current performance

On 500 uniformly sampled frames across 17 clips:
- 362 HSV candidates total (0.7/frame average)
- 110 high-confidence candidates (ref_score > 0.6)
- 94 frames with at least one likely ball (19%)

The 19% detection rate is expected for sampled frames (many are from breaks, warm-ups, ball out of water). For consecutive frames during active play, the rate should be much higher.

## What We Tried and Why We Moved On

### 1. Manual labeling with iterative CNN training

**Approach**: Web UI for labeling ball positions on sampled frames. Each "batch" of ~50 frames would:
1. Show frames with HSV candidate boxes
2. User clicks the correct candidate (or "No Ball" / "Skip")
3. Labels used to (a) retrain a MobileNetV3-Small CNN classifier, (b) tune HSV bands

**Problem**: The first batch was 85% accurate, but subsequent batches degraded. Two root causes:

- **HSV band widening**: `tune_hsv_from_labels()` only widened bands (expanded to cover observed ball HSV values), never narrowed them. Each iteration added more false positives, which made labeling harder, which produced worse training data.
- **CNN overfitting**: With <50 positive samples, the CNN couldn't generalize. The progressive unfreezing schedule (freeze backbone → unfreeze last 2 blocks → full) was sound but needed more data than manual labeling could practically provide.

**Fix applied**: Changed `tune_hsv_from_labels()` to replace bands with a tight p2–p98 fit rather than only widening. But the fundamental problem remained — manual labeling at scale was impractical and the feedback loop was unstable.

**Decision**: Abandoned iterative labeling in favor of reference-image scoring. The labeling UI and CNN training code are still in the codebase and could be useful later with more data.

### 2. GPU-accelerated detection via kornia

**Approach**: Port all pixel operations (BGR→HSV, thresholding, morphology) to GPU using PyTorch + kornia. Single GPU transfer round-trip to avoid CPU↔GPU bounce overhead.

**Problem**: On 4608×4608 frames, GPU detection was actually **slower** than CPU OpenCV:
- BGR→HSV: CPU 0.009s vs GPU 0.499s (data transfer dominates)
- Morphology with large kernels (93px pool mask dilation): GPU 0.495s vs CPU 0.3s

More critically, kornia's HSV conversion produced **numerically different results** from OpenCV's. This silently broke detection — candidate counts dropped from normal levels to near-zero.

**Decision**: Reverted to CPU OpenCV for all detection. The bottleneck is video decoding (solved by NVDEC), not per-frame pixel ops.

### 3. Wide HSV bands to compensate for NVDEC color shift

**Approach**: When we discovered NVDEC's BT.601 color shift (before fixing it), we tried widening HSV bands to accommodate the ~10-point saturation difference.

**Problem**: Wider bands caught more false positives without reliably capturing more balls. The H range grew to 10–105, spanning yellow through green to cyan. Morphological operations on these noisy masks produced unpredictable results.

**Decision**: Fixed the root cause (BT.709 conversion) instead of compensating with wider bands.

## Design Rationale

### Why reference-image scoring over CNN

1. **Zero labeling cost** — just need one picture of the ball
2. **No training instability** — the reference image is fixed, no feedback loops
3. **Interpretable** — score is a direct color similarity measure
4. **Good enough for keyframes** — we don't need per-frame ball localization at this stage; we need confident anchor points for the virtual camera

### Why wide HSV bands + scoring (instead of tight bands only)

The HSV bands are intentionally wide to maximize recall. The reference scorer then ranks candidates by ball-likeness. This two-stage approach (wide filter → discriminative scoring) is more robust than trying to find one perfect HSV range, because:
- Lighting varies between day and night games
- The ball color shifts with distance, water reflections, and equirectangular distortion
- A too-tight filter misses real balls; a too-wide filter is fine if the scorer can rank them

### Why per-frame detection rate of ~19% is acceptable

For the virtual camera path, we need confident ball positions at sparse keyframe intervals. The tracker (M3) will:
- Run detection on every frame during active play (not just sampled frames)
- Use Kalman filtering to predict ball position between detections
- Interpolate smooth camera paths through gaps (GAP_BRIDGE state)
- Fall back to tracking player activity when the ball is lost for extended periods (ACTION_PROXY state)

A 19% rate on uniformly sampled frames (including breaks and warmups) translates to a much higher rate during active play. And even during play, missing some frames is fine — the tracker only needs periodic high-confidence anchor points.

## Files

| File | Role |
|------|------|
| `src/wpv/tracking/detector.py` | HSV detection, `BallReferenceScorer`, CNN classifier, `BallDetector` |
| `scripts/label_ball.py` | Frame extraction, NVDEC decode, labeling web UI, game area masks |
| `data/ball_reference.webp` | Reference ball image (KAP7 HydroGrip) |
| `data/labeling/hsv_params.json` | Tuned HSV band parameters |
| `data/labeling/game_masks.json` | Per-clip game area polygons (17 clips) |
| `data/labeling/prep.json` | Sampled frames with candidates and ref_scores |

## Next: Ball Tracking (M3)

The detection stage is complete. Next step is the Kalman tracker + state machine that consumes per-frame detections and produces a continuous ball track for the virtual camera. See GitHub issue #4.
