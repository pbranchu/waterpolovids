# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Automated water polo video pipeline: Insta360 X5 360° footage → ball tracking → dynamic reframing → highlights → YouTube upload. Runs on a Linux server with NVIDIA GPU. No human-in-the-loop by default.

## Current State

Core pipeline is implemented: ingest, ball detection/tracking, virtual camera, render, web UI, and YouTube publish all work. Runs in Docker with `restart: unless-stopped`. Highlight extraction is the main remaining feature.

## Pipeline Architecture

1. **Ingest** (done) — format detection, manifest parsing, fisheye decode
2. **Ball detection** (done) — HSV candidates + reference-image scoring + CNN classifier
3. **Ball tracking** (done) — Kalman filter + 5-state machine (see [docs/balltracking.md](docs/balltracking.md))
4. **Virtual camera** (done) — state machine: TRACKING → SEARCH → REWIND → GAP_BRIDGE → ACTION_PROXY (see [docs/reframing.md](docs/reframing.md))
5. **Render** (done) — fisheye undistortion → 16:9 perspective reframe
6. **Web UI** (done) — game setup, upload, mask editor, batch processing (see [docs/webui.md](docs/webui.md))
7. **YouTube publish** (done) — YouTube Data API, playlist management
8. **Highlight extraction** (planned) — speed spikes, direction changes, goal proximity

## Docker deployment

The app runs in a Docker container with NVIDIA GPU access. Web UI on port 5050.

```bash
docker compose up -d            # start
docker compose logs -f          # logs
docker exec waterpolovids-wpv-1 wpv status  # CLI inside container
```

### Container path mapping

| Host | Container | Notes |
|------|-----------|-------|
| project dir | `/app/project` | bind-mount, working_dir |
| `/mnt/work/shared/Waterpolo` | `/data/raw` | source videos (read-only) |
| `/mnt/work/shared` | `/data/shared` | shared output |
| `~/.wpv` | `/secrets` | YouTube OAuth secrets (read-only) |

### Key env vars (WPV_ prefix, pydantic-settings)

- `WPV_RAW_ROOT` — where game working dirs are created (default `/data/work` in container)
- `WPV_RESULTS_DB_PATH` — SQLite DB location
- `WPV_GAME_MASKS_PATH` — path to game_masks.json
- `WPV_YOUTUBE_CLIENT_SECRETS` — path to OAuth client_secrets.json
