"""Background task runner for game processing."""

from __future__ import annotations

import json
import logging
import queue
import threading
import traceback
from pathlib import Path

from wpv.config import settings
from wpv.db import (
    get_game,
    get_game_clips,
    update_game_field,
)

log = logging.getLogger(__name__)


class TaskManager:
    """Simple background task manager with a single worker thread."""

    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue()
        self._progress: dict[str, dict] = {}
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _ensure_worker(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def enqueue(self, game_id: str) -> None:
        """Add a game to the processing queue."""
        with self._lock:
            self._progress[game_id] = {
                "status": "queued",
                "stage": None,
                "progress_pct": 0,
                "message": "Waiting in queue...",
            }
        update_game_field(game_id, "status", "queued")
        self._queue.put(game_id)
        self._ensure_worker()

    def get_progress(self, game_id: str) -> dict:
        """Get current progress for a game."""
        with self._lock:
            if game_id in self._progress:
                return self._progress[game_id].copy()
        # Fall back to DB
        game = get_game(game_id)
        if game:
            return {
                "status": game.status,
                "stage": game.current_stage,
                "progress_pct": game.progress_pct,
                "message": game.error_message or "",
            }
        return {"status": "unknown", "stage": None, "progress_pct": 0, "message": ""}

    def _update_progress(self, game_id: str, **kwargs: object) -> None:
        with self._lock:
            if game_id not in self._progress:
                self._progress[game_id] = {}
            self._progress[game_id].update(kwargs)

    def _worker(self) -> None:
        while True:
            try:
                game_id = self._queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                self._process_game(game_id)
            except Exception:
                log.exception("Error processing game %s", game_id)
                tb = traceback.format_exc()
                self._update_progress(
                    game_id, status="failed", message=tb[-500:]
                )
                update_game_field(game_id, "status", "failed")
                update_game_field(game_id, "error_message", tb[-1000:])
            finally:
                self._queue.task_done()

    def _process_game(self, game_id: str) -> None:
        from wpv.pipeline import Stage, run_pipeline

        game = get_game(game_id)
        if game is None:
            self._update_progress(game_id, status="failed", message="Game not found")
            return

        clips = get_game_clips(game_id)
        if not clips:
            self._update_progress(game_id, status="failed", message="No clips")
            update_game_field(game_id, "status", "failed")
            update_game_field(game_id, "error_message", "No clips configured")
            return

        self._update_progress(game_id, status="processing", stage="setup", progress_pct=0)
        update_game_field(game_id, "status", "processing")

        # Set up game directory
        game_dir = settings.raw_root / game_id
        game_dir.mkdir(parents=True, exist_ok=True)

        # Symlink or verify clip files exist in game directory
        for clip in clips:
            src = Path(clip.source_path)
            if not src.exists():
                self._update_progress(
                    game_id, status="failed",
                    message=f"Clip {clip.clip_index} source not found: {src}",
                )
                update_game_field(game_id, "status", "failed")
                update_game_field(game_id, "error_message", f"Missing: {src}")
                return
            dest = game_dir / src.name
            if not dest.exists():
                dest.symlink_to(src)

        # Generate game_masks.json from DB masks
        masks: dict[str, list] = {}
        for clip in clips:
            clip_key = f"clip_{clip.clip_index:03d}"
            if clip.mask_override:
                masks[clip_key] = json.loads(clip.mask_override)
            elif game.default_mask:
                masks[clip_key] = json.loads(game.default_mask)

        if masks:
            masks_path = game_dir / "game_masks.json"
            masks_path.write_text(json.dumps(masks, indent=2))

        # Generate manifest.json
        manifest_data = {
            "teams": game.name,
            "date": game.created_at[:10],
            "time": "",
            "location": "unknown",
            "clips": [],
        }
        for clip in clips:
            manifest_data["clips"].append({
                "filename": Path(clip.source_path).name,
                "duration": 0,
            })
        manifest_path = game_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_data, indent=2))

        # Stage weights for progress calculation
        stage_weights = {
            Stage.DETECT: 5,
            Stage.DECODE: 10,
            Stage.TRACK: 40,
            Stage.RENDER: 30,
            Stage.HIGHLIGHTS: 10,
            Stage.QUALITY_CHECK: 2,
            Stage.UPLOAD: 3,
        }
        total_weight = sum(stage_weights.values())
        completed_weight = 0

        def _progress(stage: Stage, msg: str) -> None:
            nonlocal completed_weight
            if msg == "done" or msg == "skipped":
                completed_weight += stage_weights.get(stage, 0)
            pct = min(99, completed_weight * 100 / total_weight)
            self._update_progress(
                game_id,
                status="processing",
                stage=stage.value,
                progress_pct=pct,
                message=f"{stage.value}: {msg}",
            )
            update_game_field(game_id, "current_stage", stage.value)
            update_game_field(game_id, "progress_pct", pct)

        result = run_pipeline(
            game_dir,
            skip_upload=not game.playlist_name,
            progress_callback=_progress,
        )

        if result.success:
            self._update_progress(
                game_id, status="completed", progress_pct=100,
                stage="done", message="Processing complete",
            )
            update_game_field(game_id, "status", "completed")
            update_game_field(game_id, "progress_pct", 100)

            # Handle playlist if configured
            if game.playlist_name and result.youtube_url:
                try:
                    from wpv.publish.youtube import (
                        add_video_to_playlist,
                        get_authenticated_service,
                        get_or_create_playlist,
                    )
                    service = get_authenticated_service()
                    pid = get_or_create_playlist(service, game.playlist_name)
                    update_game_field(game_id, "playlist_id", pid)
                    # Extract video ID from URL
                    vid_id = result.youtube_url.split("/")[-1]
                    add_video_to_playlist(service, pid, vid_id)
                except Exception:
                    log.exception("Failed to add to playlist for game %s", game_id)
        else:
            failed = [s for s in result.stages if not s.success and not s.skipped]
            err = failed[0].error if failed else "Unknown error"
            self._update_progress(
                game_id, status="failed", message=err or "Pipeline failed",
            )
            update_game_field(game_id, "status", "failed")
            update_game_field(game_id, "error_message", err)


# Module-level singleton
task_manager = TaskManager()
