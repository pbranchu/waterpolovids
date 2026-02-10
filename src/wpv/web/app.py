"""Flask app factory and all routes for the web UI."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for

from wpv.config import settings
from wpv.db import (
    GameClipRecord,
    GameRecord,
    add_game_clip,
    create_game,
    delete_game,
    get_all_games,
    get_clip_mask,
    get_game,
    get_game_clips,
    init_db,
    update_clip_field,
    update_game_field,
)


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    app.secret_key = "wpv-internal-tool"
    app.jinja_env.globals["config"] = settings

    init_db()

    # Recover games stuck as queued/processing from previous container run
    from wpv.web.tasks import task_manager
    task_manager.recover_on_startup()

    # -- Pages ---------------------------------------------------------------

    @app.route("/")
    def dashboard():
        games = get_all_games()
        return render_template("dashboard.html", games=games)

    @app.route("/game/new")
    def game_new():
        return render_template("game_new.html")

    @app.route("/game/create", methods=["POST"])
    def game_create():
        name = request.form.get("name", "").strip()
        clip_count = int(request.form.get("clip_count", 4))
        if not name:
            return redirect(url_for("game_new"))

        game_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        game = GameRecord(
            game_id=game_id,
            name=name,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        create_game(game)

        # Pre-create clip slots
        for i in range(clip_count):
            clip = GameClipRecord(game_id=game_id, clip_index=i)
            add_game_clip(clip)

        return redirect(url_for("game_upload", game_id=game_id))

    @app.route("/game/<game_id>/upload")
    def game_upload(game_id):
        game = get_game(game_id)
        if not game:
            return redirect(url_for("dashboard"))
        clips = get_game_clips(game_id)
        return render_template("game_upload.html", game=game, clips=clips)

    @app.route("/game/<game_id>/masks")
    def game_masks(game_id):
        game = get_game(game_id)
        if not game:
            return redirect(url_for("dashboard"))
        clips = get_game_clips(game_id)
        # Find first clip without a mask
        for c in clips:
            if not c.mask_override and not game.default_mask:
                return redirect(url_for("game_mask_editor", game_id=game_id, clip_idx=c.clip_index))
        # All have masks — go to first clip
        return redirect(url_for("game_mask_editor", game_id=game_id, clip_idx=0))

    @app.route("/game/<game_id>/masks/<int:clip_idx>")
    def game_mask_editor(game_id, clip_idx):
        game = get_game(game_id)
        if not game:
            return redirect(url_for("dashboard"))
        clips = get_game_clips(game_id)
        total = len(clips)

        # Get existing mask for this clip
        mask = get_clip_mask(game_id, clip_idx)
        mask_json = json.dumps(mask) if mask else "null"

        return render_template(
            "mask_editor.html",
            game=game,
            clip_idx=clip_idx,
            clip_num=clip_idx + 1,
            total_clips=total,
            progress_pct=((clip_idx + 1) / total * 100) if total else 0,
            existing_poly_json=mask_json,
        )

    @app.route("/game/<game_id>/review")
    def game_review(game_id):
        game = get_game(game_id)
        if not game:
            return redirect(url_for("dashboard"))
        clips = get_game_clips(game_id)
        clip_masks = []
        for c in clips:
            m = get_clip_mask(game_id, c.clip_index)
            clip_masks.append({"clip": c, "mask": m})
        return render_template("game_review.html", game=game, clips=clips, clip_masks=clip_masks)

    @app.route("/game/<game_id>/process", methods=["POST"])
    def game_process(game_id):
        game = get_game(game_id)
        if not game:
            return redirect(url_for("dashboard"))

        playlist = request.form.get("playlist_name", "").strip()
        if playlist:
            update_game_field(game_id, "playlist_name", playlist)

        from wpv.web.tasks import task_manager
        task_manager.enqueue(game_id)
        return redirect(url_for("dashboard"))

    # -- API endpoints -------------------------------------------------------

    @app.route("/api/upload-chunk", methods=["POST"])
    def api_upload_chunk():
        from wpv.web.uploads import handle_chunk, save_filename_hint

        game_id = request.form.get("game_id", "")
        clip_index = int(request.form.get("clip_index", 0))
        chunk_idx = int(request.form.get("chunk_idx", 0))
        total_chunks = int(request.form.get("total_chunks", 1))
        filename = request.form.get("filename", "")

        file = request.files.get("chunk")
        if not file:
            return jsonify({"ok": False, "error": "No chunk data"}), 400

        if chunk_idx == 0 and filename:
            save_filename_hint(game_id, clip_index, filename)

        result = handle_chunk(game_id, clip_index, chunk_idx, total_chunks, file.read())

        if result["complete"]:
            update_clip_field(game_id, clip_index, "source_path", result["path"])
            update_clip_field(game_id, clip_index, "filename", filename)
            update_clip_field(game_id, clip_index, "uploaded", True)
            if result["frame_path"]:
                update_clip_field(game_id, clip_index, "frame_path", result["frame_path"])

        return jsonify({"ok": True, **result})

    @app.route("/api/set-server-path", methods=["POST"])
    def api_set_server_path():
        data = request.get_json()
        game_id = data.get("game_id", "")
        clip_index = int(data.get("clip_index", 0))
        path = data.get("path", "").strip()

        if not path or not Path(path).exists():
            return jsonify({"ok": False, "error": "File not found"}), 400

        update_clip_field(game_id, clip_index, "source_path", path)
        update_clip_field(game_id, clip_index, "filename", Path(path).name)
        update_clip_field(game_id, clip_index, "uploaded", True)

        # Extract first frame
        from wpv.web.masks import extract_first_frame
        frame_dir = settings.raw_root / game_id / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frame_dir / f"clip_{clip_index:03d}.jpg"
        extract_first_frame(path, frame_path)
        if frame_path.exists():
            update_clip_field(game_id, clip_index, "frame_path", str(frame_path))

        return jsonify({"ok": True, "path": path, "frame_path": str(frame_path)})

    @app.route("/api/game/<game_id>/frame/<int:clip_idx>")
    def api_frame(game_id, clip_idx):
        clips = get_game_clips(game_id)
        for c in clips:
            if c.clip_index == clip_idx and c.frame_path:
                p = Path(c.frame_path)
                if p.exists():
                    return send_file(str(p), mimetype="image/jpeg")
        # Try extracting on-the-fly
        for c in clips:
            if c.clip_index == clip_idx and c.source_path:
                from wpv.web.masks import extract_first_frame
                frame_dir = settings.raw_root / game_id / "frames"
                frame_dir.mkdir(parents=True, exist_ok=True)
                frame_path = frame_dir / f"clip_{clip_idx:03d}.jpg"
                if extract_first_frame(c.source_path, frame_path):
                    update_clip_field(game_id, clip_idx, "frame_path", str(frame_path))
                    return send_file(str(frame_path), mimetype="image/jpeg")
        return "Frame not available", 404

    @app.route("/api/game/<game_id>/mask", methods=["POST"])
    def api_save_mask(game_id):
        data = request.get_json()
        clip_idx = int(data.get("clip_index", 0))
        polygon = data.get("polygon", [])
        apply_all = data.get("apply_all", True)

        if len(polygon) < 3:
            return jsonify({"ok": False, "error": "Need at least 3 points"}), 400

        poly_json = json.dumps(polygon)

        if apply_all:
            update_game_field(game_id, "default_mask", poly_json)
        else:
            update_clip_field(game_id, clip_idx, "mask_override", poly_json)

        # Determine next clip needing a mask
        game = get_game(game_id)
        clips = get_game_clips(game_id)
        next_clip = None
        for c in clips:
            if c.clip_index > clip_idx:
                has_mask = c.mask_override or (game and game.default_mask)
                if not has_mask:
                    next_clip = c.clip_index
                    break

        return jsonify({"ok": True, "next_clip": next_clip})

    @app.route("/api/game/<game_id>/cones/<int:clip_idx>")
    def api_cones(game_id, clip_idx):
        clips = get_game_clips(game_id)
        for c in clips:
            if c.clip_index == clip_idx and c.frame_path:
                p = Path(c.frame_path)
                if p.exists():
                    try:
                        from wpv.web.masks import detect_cone_candidates
                        markers, polygon = detect_cone_candidates(str(p))
                        return jsonify({"markers": markers, "polygon": polygon})
                    except Exception as e:
                        return jsonify({"markers": [], "polygon": [], "error": str(e)})
        return jsonify({"markers": [], "polygon": []})

    @app.route("/api/game/<game_id>/progress")
    def api_progress(game_id):
        from wpv.web.tasks import task_manager
        return jsonify(task_manager.get_progress(game_id))

    @app.route("/api/debug/worker")
    def api_debug_worker():
        from wpv.web.tasks import task_manager
        return jsonify({
            "workers_alive": sum(1 for t in task_manager._workers if t.is_alive()),
            "workers_total": len(task_manager._workers),
            "queue_size": task_manager._queue.qsize(),
            "progress": {k: v for k, v in task_manager._progress.items()},
        })

    @app.route("/api/game/<game_id>/cancel", methods=["POST"])
    def api_cancel_game(game_id):
        from wpv.web.tasks import task_manager
        was_active = task_manager.cancel(game_id)
        return jsonify({"ok": True, "was_active": was_active})

    @app.route("/api/game/<game_id>", methods=["DELETE"])
    def api_delete_game(game_id):
        game = get_game(game_id)
        if not game:
            return jsonify({"ok": False, "error": "Game not found"}), 404
        if game.status in ("processing", "queued"):
            return jsonify({"ok": False, "error": "Cannot delete while processing"}), 409
        delete_game(game_id)
        return jsonify({"ok": True})

    @app.route("/api/playlists")
    def api_playlists():
        # Collect playlist names already used/pending in the DB
        db_names = set()
        for g in get_all_games():
            if g.playlist_name:
                db_names.add(g.playlist_name)

        yt_playlists = []
        yt_titles = set()
        try:
            from wpv.publish.youtube import get_authenticated_service
            service = get_authenticated_service()
            response = service.playlists().list(
                part="snippet", mine=True, maxResults=50,
            ).execute()
            for item in response.get("items", []):
                title = item["snippet"]["title"]
                yt_playlists.append({"id": item["id"], "title": title, "source": "youtube"})
                yt_titles.add(title)
        except Exception:
            pass

        # Add pending playlists (in DB but not yet on YouTube)
        pending = [
            {"title": name, "source": "pending"}
            for name in sorted(db_names - yt_titles)
        ]
        return jsonify({"ok": True, "playlists": yt_playlists + pending})

    return app
