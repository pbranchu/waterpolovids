# waterpolovids

Automated water polo video pipeline: Insta360 X5 360° footage → ball tracking → dynamic reframing → highlights → YouTube upload.

## Pipeline

| Stage | Status | Description |
|-------|--------|-------------|
| **Ingest** | Done | Format detection, manifest parsing, stitch/decode |
| **Ball Detection** | Done | HSV candidates + reference-image scoring |
| **Ball Tracking** | Done | Kalman filter + 5-state machine (see [docs/balltracking.md](docs/balltracking.md)) |
| **Virtual Camera** | Done | State machine: TRACKING → SEARCH → REWIND → GAP_BRIDGE → ACTION_PROXY |
| **Render** | Done | Equirect → 16:9 perspective reframe (1080p–4K) |
| **Highlights** | Planned | Speed spikes, direction changes, goal proximity |
| **Web UI** | Done | Game setup, upload, mask editor, batch processing (see [docs/webui.md](docs/webui.md)) |
| **YouTube Publish** | Done | YouTube Data API upload |

Full spec: [docs/waterpolo_auto_reframe_pipeline.md](docs/waterpolo_auto_reframe_pipeline.md)

## Setup

### Docker (recommended)

```bash
docker compose up -d          # builds image, starts web UI on port 5050
docker compose logs -f        # watch logs
```

Web UI: `http://<server-ip>:5050`

The container uses `restart: unless-stopped` and starts automatically on boot.

### Local dev

```bash
pip install -e ".[dev]"
wpv ui                        # http://0.0.0.0:5000
```

Requires: Python 3.11+, NVIDIA GPU with NVDEC support, OpenCV, PyTorch.

### YouTube secrets

Place your OAuth `client_secrets.json` in `~/.wpv/client_secrets.json` (outside the repo). The container mounts `~/.wpv` at `/secrets` read-only.

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

## Docker volume mapping

| Host path | Container path | Notes |
|-----------|---------------|-------|
| `/mnt/work/gitperso/waterpolovids` | `/app/project` | Bind-mounted project dir (matches/, scripts/, DB) |
| `/mnt/work/shared/Waterpolo` | `/data/raw` | Source videos (read-only) |
| `/mnt/work/shared` | `/data/shared` | Shared output |
| `~/.wpv` | `/secrets` | YouTube OAuth secrets (read-only) |

Video paths in the web UI use container paths, e.g. `/data/raw/PRO_VID_.../video.mp4`.

## Key Directories

```
data/                          # Video files + labeling data (gitignored, symlink)
  ball_reference.webp          # Reference ball image for scoring
  labeling/
    models/ball_classifier.pth # Trained CNN model
    frames/                    # Sampled JPEG frames (4608x4608)
    prep.json                  # Frame metadata + candidates + scores
    game_masks.json            # Per-clip game area polygons
    hsv_params.json            # Tuned HSV band parameters
matches/                       # Per-game directories (gitignored)
docs/
  waterpolo_auto_reframe_pipeline.md   # Full pipeline spec
  balltracking.md                      # Ball detection/tracking design doc
  reframing.md                         # Virtual camera + render design doc
  webui.md                             # Web UI architecture + API reference
scripts/
  label_ball.py                # Detection prep + labeling web UI
src/wpv/
  tracking/detector.py         # HSV detection, reference scorer, CNN classifier
  tracking/kalman.py           # Kalman filter + 5-state tracker
  render/                      # Fisheye undistortion + perspective reframe
  web/                         # Flask web UI
  publish/youtube.py           # YouTube Data API integration
  ingest/                      # Format detection, manifest, stitch
  cli.py                       # CLI entry points
```

## CLI

```bash
wpv detect <video>              # Detect video format
wpv decode <video> -o <dir>     # Prepare equirectangular video
wpv manifest <dir>              # Parse/generate manifest
wpv label [--prep-only|--serve-only] [--count N]  # Ball labeling workflow
wpv track <match_dir>           # Run ball tracking on a match
wpv render <match_dir>          # Render reframed video
wpv ui                          # Start web UI (default: port 5000)
wpv status                      # Show pipeline status
```
