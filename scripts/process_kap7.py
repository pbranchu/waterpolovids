#!/usr/bin/env python3
"""Process all Kap7 2026 tournament videos: track, render, and copy to shared.

Usage:
    python scripts/process_kap7.py --dry-run     # show plan without executing
    python scripts/process_kap7.py               # track + render all clips
    python scripts/process_kap7.py --render-only  # skip tracking, render tracked clips
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add project root to path so wpv is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Tournament layout: clip → (game, quarter, skip?)
# ---------------------------------------------------------------------------

@dataclass
class ClipInfo:
    clip_idx: int       # 0-based: clip_000, clip_001, ...
    video_idx: int      # 1-based: PRO_VID_*_001, _002, ...
    game: int           # 1-4
    quarter: int | None # 1-4 or None (false start)
    skip: bool = False

CLIPS: list[ClipInfo] = [
    ClipInfo(0,  1,  1, 1),
    ClipInfo(1,  2,  1, 2),
    ClipInfo(2,  3,  1, 3),
    ClipInfo(3,  4,  1, None, skip=True),   # false start, 17.9s
    ClipInfo(4,  5,  1, 4),
    ClipInfo(5,  6,  2, 1),
    ClipInfo(6,  7,  2, 2),
    ClipInfo(7,  8,  2, 3),
    ClipInfo(8,  9,  2, 4),
    ClipInfo(9,  10, 3, 1),
    ClipInfo(10, 11, 3, 2),
    ClipInfo(11, 12, 3, 3),
    ClipInfo(12, 13, 3, 4),
    ClipInfo(13, 14, 4, 1),
    ClipInfo(14, 15, 4, 2),
    ClipInfo(15, 16, 4, 3),
    ClipInfo(16, 17, 4, 4),
]

SOURCE_DIR = Path("/mnt/work/shared/Waterpolo")
DATA_DIR = PROJECT_ROOT / "data"
MATCHES_DIR = PROJECT_ROOT / "matches"
GAME_MASKS_PATH = PROJECT_ROOT / "data" / "labeling" / "game_masks.json"
SHARED_OUTPUT_DIR = Path("/mnt/work/shared")

DEFAULT_PLAYLIST = "Kap7 2026"
DEFAULT_NAME_PREFIX = "Kap7 2026"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_video_path(video_idx: int) -> Path | None:
    """Find the source .mp4 for a given 1-based video index."""
    pattern = f"PRO_VID_*_{video_idx:03d}-Original"
    # Check local data/ first (may already be symlinked there)
    for d in sorted(DATA_DIR.glob(pattern)):
        mp4s = list(d.glob("PRO_VID_*.mp4"))
        if mp4s:
            return mp4s[0]
    # Fall back to shared storage
    for d in sorted(SOURCE_DIR.glob(pattern)):
        mp4s = list(d.glob("PRO_VID_*.mp4"))
        if mp4s:
            return mp4s[0]
    return None


def find_lrv_path(video_idx: int) -> Path | None:
    """Find the LRV preview file for a given 1-based video index."""
    pattern = f"PRO_VID_*_{video_idx:03d}-Original"
    for d in sorted(DATA_DIR.glob(pattern)):
        lrvs = list(d.glob("PRO_LRV_*.lrv"))
        if lrvs:
            return lrvs[0]
    for d in sorted(SOURCE_DIR.glob(pattern)):
        lrvs = list(d.glob("PRO_LRV_*.lrv"))
        if lrvs:
            return lrvs[0]
    return None


def game_match_dir(game: int) -> Path:
    return MATCHES_DIR / f"kap7-2026-game{game}"


def setup_match_dir(clip: ClipInfo) -> Path:
    """Create match dir with symlinks to source video and LRV."""
    match_dir = game_match_dir(clip.game)
    match_dir.mkdir(parents=True, exist_ok=True)
    work_dir = match_dir / "work"
    work_dir.mkdir(exist_ok=True)

    video_path = find_video_path(clip.video_idx)
    if video_path is None:
        raise FileNotFoundError(
            f"Source video not found for clip_{clip.clip_idx:03d} (video index {clip.video_idx})"
        )

    # Symlink video into match dir
    link = match_dir / video_path.name
    if not link.exists():
        link.symlink_to(video_path)

    # Symlink into work dir (needed by pipeline stages)
    work_link = work_dir / video_path.name
    if not work_link.exists():
        work_link.symlink_to(video_path)

    # Also symlink LRV if available
    lrv_path = find_lrv_path(clip.video_idx)
    if lrv_path:
        lrv_link = match_dir / lrv_path.name
        if not lrv_link.exists():
            lrv_link.symlink_to(lrv_path)

    # Copy game_masks.json to match dir if not present
    match_masks = match_dir / "game_masks.json"
    if not match_masks.exists() and GAME_MASKS_PATH.exists():
        shutil.copy2(GAME_MASKS_PATH, match_masks)

    # Migrate track JSON from old directory structure (kap7-2026-g1q1 → kap7-2026-game1)
    old_match_dir = MATCHES_DIR / "kap7-2026-g1q1"
    if old_match_dir.exists():
        old_track = old_match_dir / "work" / f"{video_path.stem}.json"
        new_track = work_dir / f"{video_path.stem}.json"
        if old_track.exists() and not new_track.exists():
            shutil.copy2(old_track, new_track)
            print(f"  Migrated track: {old_track} -> {new_track}")

    return match_dir


def track_clip(clip: ClipInfo, dry_run: bool = False) -> Path | None:
    """Track a single clip. Returns path to track JSON or None if skipped."""
    match_dir = game_match_dir(clip.game)
    work_dir = match_dir / "work"

    video_path = find_video_path(clip.video_idx)
    if video_path is None:
        print(f"  ERROR: Video not found for clip_{clip.clip_idx:03d}")
        return None

    track_json = work_dir / f"{video_path.stem}.json"
    if track_json.exists():
        print(f"  SKIP (already tracked): {track_json.name}")
        return track_json

    if dry_run:
        print(f"  WOULD TRACK: {video_path.name} -> {track_json.name}")
        return None

    clip_name = f"clip_{clip.clip_idx:03d}"
    print(f"  TRACKING: {video_path.name} ({clip_name})...")
    t0 = time.time()

    cmd = [
        "wpv", "track",
        str(video_path),
        "-o", str(track_json),
        "--clip", clip_name,
    ]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"  FAILED (exit {proc.returncode}) after {elapsed:.0f}s")
        return None

    print(f"  DONE tracking {clip_name} in {elapsed:.0f}s")
    return track_json


def render_clip(clip: ClipInfo, dry_run: bool = False) -> Path | None:
    """Render a single tracked clip. Returns path to rendered MP4 or None."""
    match_dir = game_match_dir(clip.game)
    work_dir = match_dir / "work"

    video_path = find_video_path(clip.video_idx)
    if video_path is None:
        return None

    track_json = work_dir / f"{video_path.stem}.json"
    if not track_json.exists():
        print(f"  SKIP (not tracked): {video_path.stem}")
        return None

    render_mp4 = work_dir / f"{video_path.stem}_render.mp4"
    if render_mp4.exists():
        print(f"  SKIP (already rendered): {render_mp4.name}")
        return render_mp4

    clip_name = f"clip_{clip.clip_idx:03d}"

    if dry_run:
        print(f"  WOULD RENDER: {track_json.name} -> {render_mp4.name}")
        return None

    print(f"  RENDERING: {clip_name} ({video_path.stem})...")
    t0 = time.time()

    cmd = [
        "wpv", "render",
        str(video_path),
        str(track_json),
        "-o", str(render_mp4),
        "--game-masks", str(GAME_MASKS_PATH),
        "--clip-name", clip_name,
    ]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"  FAILED render (exit {proc.returncode}) after {elapsed:.0f}s")
        return None

    print(f"  DONE rendering {clip_name} in {elapsed:.0f}s")
    return render_mp4


def register_game_in_db(
    game: int,
    game_clips: list[ClipInfo],
    playlist_name: str,
    name_prefix: str,
) -> None:
    """Register a game and its clips in the web UI database."""
    from datetime import datetime, timezone

    from wpv.db import (
        GameClipRecord,
        GameRecord,
        add_game_clip,
        create_game,
        get_game,
        init_db,
    )

    init_db()

    game_id = f"kap7-2026-game{game}"
    game_name = f"{name_prefix} Game {game}"

    # Skip if already registered
    existing = get_game(game_id)
    if existing is not None:
        print(f"  DB: Game {game_id} already registered (status={existing.status})")
        return

    # Load default mask from game_masks.json (use first clip's mask)
    default_mask = None
    if GAME_MASKS_PATH.exists():
        with open(GAME_MASKS_PATH) as f:
            all_masks = json.load(f)
        first_clip = game_clips[0]
        clip_key = f"clip_{first_clip.clip_idx:03d}"
        if clip_key in all_masks:
            default_mask = json.dumps(all_masks[clip_key])

    record = GameRecord(
        game_id=game_id,
        name=game_name,
        created_at=datetime.now(timezone.utc).isoformat(),
        status="setup",
        playlist_name=playlist_name,
        default_mask=default_mask,
    )
    create_game(record)
    print(f"  DB: Created game {game_id} ({game_name})")

    # Register clips (0-based within game)
    for clip_idx_in_game, clip in enumerate(game_clips):
        video_path = find_video_path(clip.video_idx)
        filename = video_path.name if video_path else ""
        source = str(video_path) if video_path else ""

        # Look up per-clip mask override from game_masks.json
        global_clip_key = f"clip_{clip.clip_idx:03d}"
        mask_override = None
        if GAME_MASKS_PATH.exists():
            with open(GAME_MASKS_PATH) as f:
                all_masks = json.load(f)
            if global_clip_key in all_masks:
                mask_override = json.dumps(all_masks[global_clip_key])

        clip_record = GameClipRecord(
            game_id=game_id,
            clip_index=clip_idx_in_game,
            filename=filename,
            source_path=source,
            uploaded=True,  # source files already on disk
            mask_override=mask_override,
        )
        add_game_clip(clip_record)
        q_label = f"Q{clip.quarter}" if clip.quarter else "?"
        print(f"    Clip {clip_idx_in_game}: {filename} ({q_label})")


def update_game_progress(
    game: int,
    stage: str,
    progress_pct: float,
    status: str = "processing",
) -> None:
    """Update game progress in the web UI database."""
    from wpv.db import get_game, init_db, update_game_field

    init_db()
    game_id = f"kap7-2026-game{game}"
    if get_game(game_id) is None:
        return
    update_game_field(game_id, "current_stage", stage)
    update_game_field(game_id, "progress_pct", progress_pct)
    update_game_field(game_id, "status", status)


def copy_to_shared(clip: ClipInfo, render_mp4: Path) -> Path:
    """Copy rendered clip to shared storage with descriptive name."""
    q_label = f"Quarter {clip.quarter}" if clip.quarter else "Extra"
    dest_name = f"Kap7 2026 Game {clip.game} {q_label}.mp4"
    dest = SHARED_OUTPUT_DIR / dest_name
    print(f"  COPY: {render_mp4.name} -> {dest}")
    shutil.copy2(render_mp4, dest)
    return dest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Process Kap7 2026 tournament videos")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--render-only", action="store_true", help="Skip tracking, render only")
    parser.add_argument("--copy-only", action="store_true", help="Only copy renders to shared")
    parser.add_argument("--game", type=int, help="Process only this game number (1-4)")
    parser.add_argument("--playlist", default=DEFAULT_PLAYLIST, help="Playlist name for all games")
    parser.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX, help="Prefix for game names")
    args = parser.parse_args()

    active_clips = [c for c in CLIPS if not c.skip]
    if args.game:
        active_clips = [c for c in active_clips if c.game == args.game]

    print(f"Kap7 2026 Tournament Processing")
    print(f"================================")
    print(f"Active clips: {len(active_clips)} (skipping {sum(1 for c in CLIPS if c.skip)} false starts)")
    print()

    # Show inventory
    for c in active_clips:
        vp = find_video_path(c.video_idx)
        status = "FOUND" if vp else "MISSING"
        q_label = f"Q{c.quarter}" if c.quarter else "skip"
        print(f"  clip_{c.clip_idx:03d} (idx {c.video_idx:03d}) Game {c.game} {q_label:>4}  [{status}]")
    print()

    # Setup match directories
    print("Setting up match directories...")
    for c in active_clips:
        if not args.dry_run:
            setup_match_dir(c)
        else:
            print(f"  WOULD CREATE: {game_match_dir(c.game)}")

    # Deduplicate game dirs for display
    games_seen = set()
    for c in active_clips:
        if c.game not in games_seen:
            games_seen.add(c.game)
            print(f"  {game_match_dir(c.game)}")
    print()

    # Register games in web UI database
    print("Registering games in web UI database...")
    games_clips: dict[int, list[ClipInfo]] = {}
    for c in active_clips:
        games_clips.setdefault(c.game, []).append(c)
    for game_num in sorted(games_clips):
        if not args.dry_run:
            register_game_in_db(
                game_num, games_clips[game_num],
                playlist_name=args.playlist,
                name_prefix=args.name_prefix,
            )
        else:
            game_id = f"kap7-2026-game{game_num}"
            print(f"  WOULD REGISTER: {game_id} ({len(games_clips[game_num])} clips)")
    print()

    # Track
    if not args.render_only and not args.copy_only:
        print("Tracking clips...")
        for i, c in enumerate(active_clips, 1):
            q_label = f"Q{c.quarter}" if c.quarter else "?"
            print(f"[{i}/{len(active_clips)}] Game {c.game} {q_label} (clip_{c.clip_idx:03d}):")
            if not args.dry_run:
                pct = (i - 1) / len(active_clips) * 50  # tracking = first 50%
                update_game_progress(c.game, "tracking", pct)
            track_clip(c, dry_run=args.dry_run)
        print()

    # Render
    if not args.copy_only:
        print("Rendering clips...")
        for i, c in enumerate(active_clips, 1):
            q_label = f"Q{c.quarter}" if c.quarter else "?"
            print(f"[{i}/{len(active_clips)}] Game {c.game} {q_label} (clip_{c.clip_idx:03d}):")
            if not args.dry_run:
                pct = 50 + (i - 1) / len(active_clips) * 50  # render = second 50%
                update_game_progress(c.game, "rendering", pct)
            render_clip(c, dry_run=args.dry_run)
        print()

    # Copy to shared
    print("Copying to shared storage...")
    for c in active_clips:
        match_dir = game_match_dir(c.game)
        work_dir = match_dir / "work"
        video_path = find_video_path(c.video_idx)
        if video_path is None:
            continue
        render_mp4 = work_dir / f"{video_path.stem}_render.mp4"
        if render_mp4.exists():
            if args.dry_run:
                q_label = f"Quarter {c.quarter}" if c.quarter else "Extra"
                print(f"  WOULD COPY: {render_mp4.name} -> Kap7 2026 Game {c.game} {q_label}.mp4")
            else:
                copy_to_shared(c, render_mp4)
        else:
            q_label = f"Q{c.quarter}" if c.quarter else "?"
            print(f"  SKIP (not rendered): Game {c.game} {q_label}")

    # Mark completed games in DB
    if not args.dry_run:
        for game_num in sorted(games_clips):
            update_game_progress(game_num, "completed", 100.0, status="completed")
            print(f"  DB: Game kap7-2026-game{game_num} marked completed")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
