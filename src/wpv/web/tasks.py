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
    """Background task manager with parallel game workers.

    Multiple worker threads pull games from a shared queue, so several
    games can be processed concurrently (e.g. one rendering while another
    tracks).  The number of workers is controlled by settings.parallel_games.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue()
        self._progress: dict[str, dict] = {}
        self._workers: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._cancelled: set[str] = set()

    def recover_on_startup(self) -> None:
        """Re-enqueue games stuck as queued/processing from a previous run."""
        from wpv.db import get_all_games
        for game in get_all_games():
            if game.status in ("queued", "processing"):
                log.info("Recovering stuck game %s (was %s)", game.game_id, game.status)
                self._queue.put(game.game_id)
                with self._lock:
                    self._progress[game.game_id] = {
                        "status": "queued",
                        "stage": None,
                        "progress_pct": 0,
                        "message": "Re-queued after restart",
                    }
                update_game_field(game.game_id, "status", "queued")
        if not self._queue.empty():
            self._ensure_workers()

    def _ensure_workers(self) -> None:
        """Start worker threads up to parallel_games, replacing any dead ones."""
        # Clean out dead threads
        self._workers = [t for t in self._workers if t.is_alive()]

        # Re-enqueue orphaned games if all workers died and queue is empty
        if not self._workers and self._queue.empty():
            from wpv.db import get_all_games
            for game in get_all_games():
                if game.status in ("queued", "processing"):
                    already_queued = False
                    with self._lock:
                        already_queued = game.game_id in self._progress and \
                            self._progress[game.game_id].get("status") in ("queued", "processing")
                    if not already_queued:
                        log.info("Re-enqueuing orphaned game %s", game.game_id)
                        self._queue.put(game.game_id)
                        with self._lock:
                            self._progress[game.game_id] = {
                                "status": "queued", "stage": None,
                                "progress_pct": 0, "message": "Re-queued (worker restart)",
                            }

        target = settings.parallel_games
        while len(self._workers) < target:
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self._workers.append(t)

    def enqueue(self, game_id: str) -> None:
        """Add a game to the processing queue."""
        with self._lock:
            self._cancelled.discard(game_id)
            self._progress[game_id] = {
                "status": "queued",
                "stage": None,
                "progress_pct": 0,
                "message": "Waiting in queue...",
            }
        update_game_field(game_id, "status", "queued")
        self._queue.put(game_id)
        self._ensure_workers()

    def cancel(self, game_id: str) -> bool:
        """Cancel a queued or processing game. Returns True if it was active."""
        with self._lock:
            was_active = game_id in self._progress and self._progress[game_id].get("status") in ("queued", "processing")
            self._cancelled.add(game_id)
            self._progress[game_id] = {
                "status": "cancelled",
                "stage": None,
                "progress_pct": 0,
                "message": "Cancelled by user",
            }
        update_game_field(game_id, "status", "cancelled")
        return was_active

    def is_cancelled(self, game_id: str) -> bool:
        """Check if a game has been cancelled."""
        with self._lock:
            return game_id in self._cancelled

    def get_progress(self, game_id: str) -> dict:
        """Get current progress for a game."""
        # Restart workers if they all died
        alive = any(t.is_alive() for t in self._workers)
        if not alive:
            self._ensure_workers()
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
                game_id = self._queue.get(timeout=5)
            except queue.Empty:
                # Exit if queue has been empty for a while (will be restarted if needed)
                continue
            try:
                # Skip if cancelled while queued
                if self.is_cancelled(game_id):
                    log.info("Skipping cancelled game %s", game_id)
                    continue
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
                # Clean up cancel flag
                with self._lock:
                    self._cancelled.discard(game_id)

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

        import re
        import time as _time
        _last_db_write = [0.0]

        def _progress(stage: Stage, msg: str) -> None:
            nonlocal completed_weight
            # Check for cancellation between stages
            if self.is_cancelled(game_id):
                raise _CancelledError(game_id)

            if msg == "done" or msg == "skipped":
                completed_weight += stage_weights.get(stage, 0)
            # For intra-stage messages containing a percentage, interpolate
            intra_pct = 0.0
            if msg not in ("starting", "done", "skipped") and not msg.startswith("failed"):
                m = re.search(r'\((\d+)%\)', msg)
                if m:
                    intra_pct = int(m.group(1)) / 100.0
            base_pct = completed_weight * 100 / total_weight
            stage_pct = stage_weights.get(stage, 0) * intra_pct * 100 / total_weight
            pct = min(99, base_pct + stage_pct)
            # Always update in-memory (cheap, serves API instantly)
            self._update_progress(
                game_id,
                status="processing",
                stage=stage.value,
                progress_pct=pct,
                message=f"{stage.value}: {msg}",
            )
            # Throttle DB writes to at most once per 5 seconds
            now = _time.monotonic()
            if now - _last_db_write[0] >= 5 or msg in ("starting", "done", "skipped") or msg.startswith("failed"):
                update_game_field(game_id, "current_stage", stage.value)
                update_game_field(game_id, "progress_pct", pct)
                _last_db_write[0] = now

        try:
            result = run_pipeline(
                game_dir,
                skip_upload=not game.playlist_name,
                progress_callback=_progress,
            )
        except _CancelledError:
            log.info("Game %s cancelled during processing", game_id)
            self._update_progress(
                game_id, status="cancelled", message="Cancelled by user",
            )
            update_game_field(game_id, "status", "cancelled")
            return

        if result.success:
            self._update_progress(
                game_id, status="completed", progress_pct=100,
                stage="done", message="Processing complete",
            )
            update_game_field(game_id, "status", "completed")
            update_game_field(game_id, "progress_pct", 100)

            # Handle playlist if configured — add ALL uploaded videos
            if game.playlist_name and result.youtube_urls:
                try:
                    from wpv.publish.youtube import (
                        add_video_to_playlist,
                        get_authenticated_service,
                        get_or_create_playlist,
                    )
                    service = get_authenticated_service()
                    pid = get_or_create_playlist(service, game.playlist_name)
                    update_game_field(game_id, "playlist_id", pid)
                    for url in result.youtube_urls:
                        vid_id = url.split("/")[-1]
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


class _CancelledError(Exception):
    """Raised inside progress callback when a game is cancelled."""
    pass


# Module-level singleton
task_manager = TaskManager()
