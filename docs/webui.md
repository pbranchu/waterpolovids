# Web UI for Game Setup and Processing

## Overview

The web UI (`wpv ui`) is an internal tool for managing the full lifecycle of a water polo game video: uploading clips, setting pool boundary masks, choosing a YouTube playlist, and kicking off background processing. It replaces the manual CLI workflow with a guided wizard flow.

Runs on Flask, served from a single process with a background daemon thread for pipeline execution. Designed for LAN use on the processing server — no authentication, no public exposure.

## Quick Start

```bash
wpv ui                        # http://0.0.0.0:5000
wpv ui --port 8080            # custom port
wpv ui --host 127.0.0.1       # localhost only
```

## Architecture

```
src/wpv/web/
├── __init__.py          # exports create_app
├── app.py               # Flask app factory, all routes
├── tasks.py             # background task runner (daemon thread + queue)
├── masks.py             # cone detection + first-frame extraction
├── uploads.py           # chunked file upload handler
├── templates/
│   ├── base.html        # dark theme base layout
│   ├── dashboard.html   # game list with status/progress
│   ├── game_new.html    # create game form
│   ├── game_upload.html # per-clip upload interface
│   ├── mask_editor.html # polygon boundary editor
│   └── game_review.html # summary + playlist + process trigger
└── static/
    ├── style.css        # shared dark theme (#1a1a2e / #16213e)
    ├── dashboard.js     # polls progress for active games
    ├── upload.js        # chunked upload + drag-drop
    └── mask-editor.js   # polygon drawing, zoom inset, cone snap
```

All routes live in `app.py` — no blueprints, since this is an internal single-purpose tool.

## User Flow

### 1. Dashboard (`GET /`)

Lists all games with status badges, progress bars, and action links. Games in `processing` or `queued` state auto-poll for progress every 3 seconds via `dashboard.js`.

Status lifecycle: `setup` → `queued` → `processing` → `completed` | `failed`

### 2. New Game (`GET /game/new`)

Form with two fields:
- **Game Name** — free text, used as the video title on YouTube (e.g. "Sharks vs Dolphins - Jan 15")
- **Clip Count** — how many video files make up this game (default: 4)

Creates a `GameRecord` in the database and pre-creates empty `GameClipRecord` slots.

### 3. Upload Clips (`GET /game/<id>/upload`)

Per-clip cards with two input methods:

- **Browser upload** — drag-and-drop or file picker. Files are uploaded in 10 MB chunks via `POST /api/upload-chunk`. Assembled on completion into `{raw_root}/{game_id}/`. Progress bar per clip.
- **Server path** — paste an absolute path to a video already on the server. No copy is made; the path is recorded as-is and symlinked into the game directory at processing time.

After upload/path-set, the first frame is extracted automatically (`masks.py:extract_first_frame`) for the mask editor.

### 4. Set Masks (`GET /game/<id>/masks/<clip_idx>`)

Polygon editor for marking pool boundaries on each clip's first frame. Adapted from the labeling UI in `scripts/label_ball.py`.

**Features:**
- Click to place vertices on the pool boundary
- **Zoom inset** — 8x magnified view follows the cursor for precise placement
- **Cone detection** — press `H` to highlight detected yellow/red cone markers; clicks snap to the nearest cone within 30px
- **Auto-detection** — if cone detection finds a valid polygon, it's pre-filled
- **"Apply to all clips" checkbox** (default: checked) — saves the polygon as the game-level `default_mask`, applied to all clips. Uncheck to save a per-clip override.
- Keyboard: `Z` undo, `C` clear, `H` toggle cones, `Enter` save, `←`/`→` navigate clips

**Mask storage:** The `games` table has a `default_mask` column (JSON polygon). Each clip in `game_clips` has a `mask_override` column (NULL = use game default). `get_clip_mask()` resolves the effective mask.

### 5. Review & Process (`GET /game/<id>/review`)

Summary page showing:
- Thumbnail + mask status for each clip
- **Playlist name** input — type a name or click an existing YouTube playlist. The playlist is created automatically if it doesn't exist. Multiple games with the same playlist name share one YouTube playlist (e.g. "Summer Tournament 2025").
- **"Start Processing" button** — enqueues the game and redirects to the dashboard

Leave the playlist field blank to skip YouTube upload entirely.

## Processing Pipeline

`POST /game/<id>/process` enqueues the game into `TaskManager` (a `queue.Queue` + daemon thread in `tasks.py`). The worker:

1. Reads game + clips from the database
2. Creates the game directory under `{raw_root}/{game_id}/`
3. Symlinks video files into the game directory
4. Generates `game_masks.json` from DB masks (default + overrides)
5. Generates `manifest.json` from game metadata
6. Calls `run_pipeline()` with a progress callback
7. On success with a playlist configured: `get_or_create_playlist()` + `add_video_to_playlist()`
8. Updates DB status to `completed` or `failed`

Progress is tracked in-memory (for fast polling) and persisted to the DB (for page reloads). Stage weights for the progress bar:

| Stage         | Weight |
|---------------|--------|
| detect        | 5      |
| decode        | 10     |
| track         | 40     |
| render        | 30     |
| highlights    | 10     |
| quality-check | 2      |
| upload        | 3      |

The dashboard polls `GET /api/game/<id>/progress` every 3 seconds for active games and updates the progress bar and status badge in-place.

## Database Schema

Two new tables added to `db.py` alongside the existing `results` table:

```sql
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'setup',
    playlist_name TEXT,
    playlist_id TEXT,
    default_mask TEXT,          -- JSON polygon, applied to all clips
    error_message TEXT,
    current_stage TEXT,
    progress_pct REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS game_clips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL REFERENCES games(game_id),
    clip_index INTEGER NOT NULL,
    filename TEXT NOT NULL DEFAULT '',
    source_path TEXT NOT NULL DEFAULT '',
    uploaded INTEGER NOT NULL DEFAULT 0,
    mask_override TEXT,         -- JSON polygon, NULL = use game default
    frame_path TEXT,
    UNIQUE(game_id, clip_index)
);
```

Dataclasses: `GameRecord`, `GameClipRecord`. CRUD functions: `create_game`, `get_game`, `get_all_games`, `update_game_field`, `add_game_clip`, `get_game_clips`, `update_clip_field`, `get_clip_mask`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/upload-chunk` | Chunked file upload (form: `game_id`, `clip_index`, `chunk_idx`, `total_chunks`, `filename`, `chunk`) |
| `POST` | `/api/set-server-path` | Set clip from existing server path (JSON: `game_id`, `clip_index`, `path`) |
| `GET`  | `/api/game/<id>/frame/<clip_idx>` | Serve first-frame JPEG for mask canvas |
| `POST` | `/api/game/<id>/mask` | Save polygon (JSON: `clip_index`, `polygon`, `apply_all`) |
| `GET`  | `/api/game/<id>/cones/<clip_idx>` | Cone detection results (JSON: `markers`, `polygon`) |
| `GET`  | `/api/game/<id>/progress` | Processing progress (JSON: `status`, `stage`, `progress_pct`, `message`) |
| `GET`  | `/api/playlists` | List authenticated user's YouTube playlists |

## Configuration

Two new settings in `config.py` (env vars `WPV_WEB_UPLOAD_CHUNK_SIZE_MB`, `WPV_WEB_PORT`):

| Setting | Default | Description |
|---------|---------|-------------|
| `web_upload_chunk_size_mb` | 10 | Upload chunk size in MB |
| `web_port` | 5000 | Default HTTP port |

## Dependencies

Flask was moved from `dev` to main dependencies in `pyproject.toml`.
