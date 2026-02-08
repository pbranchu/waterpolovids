# waterpolovids

Automated water polo video pipeline: Insta360 X5 360° footage → ball tracking → dynamic reframing → highlights → YouTube upload.

## Pipeline

| Stage | Status | Description |
|-------|--------|-------------|
| **Ingest** | Done | Format detection, manifest parsing, stitch/decode |
| **Ball Detection** | Done | HSV candidates + reference-image scoring |
| **Ball Tracking** | Next | Kalman filter + state machine (see [docs/balltracking.md](docs/balltracking.md)) |
| **Virtual Camera** | Planned | State machine: TRACKING → SEARCH → REWIND → GAP_BRIDGE → ACTION_PROXY |
| **Render** | Planned | Equirect → 16:9 perspective reframe (1080p–4K) |
| **Highlights** | Planned | Speed spikes, direction changes, goal proximity |
| **YouTube Publish** | Planned | YouTube Data API upload |

Full spec: [docs/waterpolo_auto_reframe_pipeline.md](docs/waterpolo_auto_reframe_pipeline.md)

## Setup

```bash
pip install -e ".[dev]"
```

Requires: Python 3.11+, NVIDIA GPU with NVDEC support, OpenCV, PyTorch.

Optional for hardware-accelerated decode: `PyNvVideoCodec`.

## Ball Detection

The detector uses a two-stage approach:

1. **HSV candidate generation** — wide-band color thresholding + morphology + circularity filtering, masked to pool/game area
2. **Reference-image scoring** — candidates scored by color similarity to a known ball image (KAP7 HydroGrip) via HS histogram back-projection

No manual labeling required. See [docs/balltracking.md](docs/balltracking.md) for details on why this approach was chosen.

### Quick test

```bash
# Prep: extract 500 sampled frames, detect candidates, score them
python scripts/label_ball.py prep --count 500

# Review game area masks (optional)
python scripts/label_ball.py batch
# → visit http://localhost:5001/masks
```

## Key Directories

```
data/                          # Video files + labeling data (gitignored)
  ball_reference.webp          # Reference ball image for scoring
  labeling/
    frames/                    # Sampled JPEG frames (4608x4608)
    prep.json                  # Frame metadata + candidates + scores
    game_masks.json            # Per-clip game area polygons
    hsv_params.json            # Tuned HSV band parameters
docs/
  waterpolo_auto_reframe_pipeline.md   # Full pipeline spec
  balltracking.md                      # Ball detection/tracking design doc
scripts/
  label_ball.py                # Detection prep + labeling web UI
src/wpv/
  tracking/detector.py         # HSV detection, reference scorer, CNN classifier
  ingest/                      # Format detection, manifest, stitch
  cli.py                       # CLI entry points
```

## CLI

```bash
wpv detect <video>              # Detect video format
wpv decode <video> -o <dir>     # Prepare equirectangular video
wpv manifest <dir>              # Parse/generate manifest
wpv label [--prep-only|--serve-only] [--count N]  # Ball labeling workflow
```
