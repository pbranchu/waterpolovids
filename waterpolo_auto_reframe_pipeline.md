
# Water Polo Auto-Reframe + Highlights Pipeline
## Insta360 X5 → Linux Server → YouTube

This document is a **complete technical specification** for an automated, no-human-in-the-loop (or minimal labeling) system that:
- ingests Insta360 X5 footage,
- tracks a water polo ball from a distant camera,
- performs dynamic FOV reframing and zoom,
- generates highlights,
- optionally applies AI upscaling,
- publishes results to YouTube.

This file is written so an AI system or engineering team can directly implement it.

---

## 1. Capture Strategy

### 1.1 Camera Mode
**Primary mode: Full 360° capture**

Rationale:
- Enables post-production reframing in any direction.
- Allows aggressive digital zoom while preserving context.
- Essential for recovering action after occlusion or rapid ball movement.

Single-lens mode may be used for non-game content but is not recommended for automated ball-follow reframing.

### 1.2 Camera Placement
- High and centered relative to pool
- Fixed mount (no physical panning)
- Stable exposure (avoid aggressive auto-exposure if possible)

---

## 2. Ingest (No Cloud)

### 2.1 Folder Layout
Raw files are pushed directly to the server:

```
/ingest/incoming/<match_id>/
  ├── RAW/
  │   ├── *.insv
  │   └── *.lrv (optional)
  ├── manifest.json
  └── UPLOAD_DONE
```

### 2.2 Required Metadata (`manifest.json`)
```json
{
  "match_id": "2026-02-05_St-Francis-Pool_U16_SJA-vs-XYZ",
  "match_date": "2026-02-05",
  "start_time_local": "15:30:00",
  "timezone": "America/Los_Angeles",
  "location_name": "St Francis Pool",
  "teams": "SJA vs XYZ",
  "camera_id": "X5-001",
  "operator": "Name"
}
```

The pipeline must not begin until `UPLOAD_DONE` exists.

---

## 3. Orchestration

- Use **Airflow or Prefect**
- Job state stored in **Postgres**
- Each stage is idempotent
- RAW inputs are immutable

Pipeline stages:
1. Validate ingest
2. Stitch/decode
3. Ball tracking
4. Virtual camera generation
5. Render reframed video
6. Highlight extraction
7. Optional AI upscaling
8. Publish to YouTube

---

## 4. Decode & Stitch

### 4.1 Tooling
- Use **Insta360 Media SDK (Linux)** if available
- GPU recommended (NVIDIA)

### 4.2 Output
Generate a master equirectangular intermediate:

```
/work/<match_id>/intermediate/equirect_master.mp4
```

This is the only source used for tracking and rendering.

---

## 5. Ball Tracking System

### 5.1 Key Assumptions
- Ball color and shape are invariant
- Lighting, scale, occlusion vary
- Camera is far from pool; ball may be only a few pixels

### 5.2 Detection Architecture (Hybrid)

#### Stage A — Candidate Generation (Fast, Deterministic)
Per frame:
- HSV yellow threshold (multiple lighting bands)
- Morphological cleanup
- Connected components
- Filters:
  - size (pixel bounds)
  - circularity
  - saturation/value
  - pool ROI constraint

Outputs many candidates, including false positives.

#### Stage B — Candidate Verification
For each candidate crop:
- Lightweight CNN classifier or detector
- Outputs confidence score

Only verified candidates enter tracking.

---

## 6. Tracking State Machine

### 6.1 States
- TRACKING
- SEARCH_FORWARD
- REWIND_BACKWARD
- GAP_BRIDGE
- ACTION_PROXY (fallback)

### 6.2 TRACKING
- Detector runs every frame or every N frames
- Kalman filter maintains position & velocity
- Track confidence maintained

**Loss condition (point_1):**
- No valid detection for L frames (e.g. 0.75s)

---

### 6.3 SEARCH_FORWARD (Skip-Ahead)
Used after loss.

Algorithm:
- Jump forward by Δ seconds (default 5s)
- Sample short window (e.g. 10 frames)
- If ball not found, jump forward again
- Stop when:
  - Ball is reliably reacquired, or
  - Max gap exceeded (e.g. 45s)

**Reacquire rule:**
- High confidence
- Persistent for P frames
- Plausible motion continuity

Reacquisition time = `t_found`

---

### 6.4 REWIND_BACKWARD
Find where the ball *first* becomes visible before `t_found`.

Algorithm:
1. Step backward in coarse increments (e.g. 0.25s)
2. When detection fails, refine with frame-level search
3. Output `point_2` (last visible frame)

---

### 6.5 GAP_BRIDGE (Occlusion Handling)
Bridge camera motion between:
- `point_1` (last known before loss)
- `point_2` (first known after reappearance)

Rules:
- Interpolate yaw/pitch with ease-in/ease-out
- Clamp angular velocity
- Gradually widen FOV
- Optional guidance from action proxy

This produces smooth camera motion even when ball is hidden.

---

## 7. Action Proxy (Fallback)
When ball is lost too long:
- Compute center of motion (players cluster)
- Track goal-area activity if available
- Reframe to proxy with wider FOV
- Resume ball tracking when available

---

## 8. Virtual Camera Path

Tracking outputs a timeline:

```
t → yaw, pitch, fov, confidence, mode
```

Constraints:
- Max angular velocity
- Confidence-driven zoom
- Horizon stabilization

---

## 9. Rendering

Render reframed video from equirect master:
- Output: 16:9
- Resolution: 1080p / 1440p / 4K
- Codec: H.264 or H.265

Output path:
```
/output/<match_id>/full_game_reframed.mp4
```

---

## 10. Highlight Generation

Signals:
- Ball speed spikes
- Direction changes
- Goal-area proximity
- Audio peaks (optional)

Process:
- Generate candidate segments
- Rank by intensity + novelty
- Merge overlaps
- Export 3–8 min montage

Output:
```
/output/<match_id>/highlights.mp4
```

---

## 11. AI Upscaling (Optional)

Placement:
- After final render
- Before upload

Purpose:
- Improve perceived quality after heavy digital zoom
- Cosmetic only (not used for detection)

---

## 12. Quality Gates (No-Human Safety)

Before upload:
- Track coverage %
- Max angular velocity
- Zoom stability
- Minimum sharpness threshold

If failed:
- Fall back to wide reframed version
- Or publish highlights only

---

## 13. Publishing to YouTube

- Use YouTube Data API
- Title/description from manifest
- Store video IDs + URLs in database

---

## 14. Versioning

Version everything:
- Detection model
- Tracking parameters
- SDK version
- Render profiles

---

## 15. Summary

This pipeline:
- Avoids Insta360 Studio limitations
- Handles occlusion explicitly
- Exploits ball invariance
- Produces broadcast-style output
- Scales to unattended operation
