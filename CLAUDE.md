# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Automated water polo video pipeline: Insta360 X5 360° footage → ball tracking → dynamic reframing → highlights → YouTube upload. Runs on a Linux server with NVIDIA GPU. No human-in-the-loop by default.

## Current State

This repository is in the **design phase**. The full technical specification lives in `docs/waterpolo_auto_reframe_pipeline.md`. No code has been implemented yet.

## Pipeline Architecture (from spec)

Eight sequential stages, each idempotent, orchestrated by Airflow or Prefect with Postgres state:

1. **Ingest validation** — wait for `UPLOAD_DONE` sentinel, parse `manifest.json`
2. **Stitch/decode** — Insta360 Media SDK → equirectangular master MP4
3. **Ball tracking** — HSV candidate generation + CNN verification, Kalman filter
4. **Virtual camera path** — state machine (TRACKING → SEARCH_FORWARD → REWIND_BACKWARD → GAP_BRIDGE → ACTION_PROXY)
5. **Render** — equirect → 16:9 reframed video (1080p–4K, H.264/H.265)
6. **Highlight extraction** — speed spikes, direction changes, goal proximity, audio peaks
7. **AI upscaling** (optional) — post-render, pre-upload
8. **YouTube publish** — via YouTube Data API

### Key directory conventions (from spec)

- `/ingest/incoming/<match_id>/RAW/` — raw `.insv` files + `manifest.json` + `UPLOAD_DONE`
- `/work/<match_id>/intermediate/` — equirectangular master
- `/output/<match_id>/` — `full_game_reframed.mp4`, `highlights.mp4`
