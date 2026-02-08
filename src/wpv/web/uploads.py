"""Chunked file upload handling for the web UI."""

from __future__ import annotations

import shutil
from pathlib import Path

from wpv.config import settings
from wpv.web.masks import extract_first_frame


def _upload_dir() -> Path:
    return settings.raw_root / "_uploads"


def handle_chunk(
    game_id: str,
    clip_index: int,
    chunk_idx: int,
    total_chunks: int,
    data: bytes,
) -> dict:
    """Handle a single chunk of a file upload.

    Returns dict with keys: complete (bool), path (str|None), frame_path (str|None).
    """
    upload_tmp = _upload_dir() / game_id / f"clip_{clip_index:03d}"
    upload_tmp.mkdir(parents=True, exist_ok=True)

    chunk_path = upload_tmp / f"chunk_{chunk_idx:06d}"
    chunk_path.write_bytes(data)

    # Check if all chunks are present
    existing = list(upload_tmp.glob("chunk_*"))
    if len(existing) < total_chunks:
        return {"complete": False, "path": None, "frame_path": None}

    # Assemble chunks
    dest_dir = settings.raw_root / game_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Read the original filename from metadata if present, else use clip index
    meta_file = upload_tmp / "filename.txt"
    if meta_file.exists():
        filename = meta_file.read_text().strip()
    else:
        filename = f"clip_{clip_index:03d}.mp4"

    dest_path = dest_dir / filename
    with open(dest_path, "wb") as out:
        for i in range(total_chunks):
            cp = upload_tmp / f"chunk_{i:06d}"
            out.write(cp.read_bytes())

    # Cleanup temp chunks
    shutil.rmtree(str(upload_tmp))

    # Extract first frame
    frame_dir = dest_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_path = frame_dir / f"clip_{clip_index:03d}.jpg"
    extract_first_frame(dest_path, frame_path)

    return {
        "complete": True,
        "path": str(dest_path),
        "frame_path": str(frame_path) if frame_path.exists() else None,
    }


def save_filename_hint(game_id: str, clip_index: int, filename: str) -> None:
    """Store original filename so it can be used during assembly."""
    upload_tmp = _upload_dir() / game_id / f"clip_{clip_index:03d}"
    upload_tmp.mkdir(parents=True, exist_ok=True)
    (upload_tmp / "filename.txt").write_text(filename)
