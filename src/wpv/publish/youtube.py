"""Upload reframed video and highlights to YouTube via Data API v3."""

from __future__ import annotations

import http.client
import random
import time
from dataclasses import dataclass
from pathlib import Path

from wpv.config import settings
from wpv.ingest.manifest import MatchManifest


@dataclass
class UploadMetadata:
    title: str
    description: str
    tags: list[str]
    privacy: str = "unlisted"
    category_id: str = "17"  # Sports


@dataclass
class UploadResult:
    video_id: str
    url: str


# -- OAuth2 helpers ----------------------------------------------------------

_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
]

MAX_RETRIES = 10
RETRIABLE_STATUS_CODES = [500, 502, 503, 504]


def get_authenticated_service(
    client_secrets: Path | None = None,
    token_path: Path | None = None,
):
    """Return an authenticated YouTube Data API v3 service.

    If a stored token exists and is valid, reuse it.
    Otherwise run ``InstalledAppFlow.run_local_server(port=0)`` for
    browser-based consent and persist the refresh token.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    secrets = client_secrets or settings.youtube_client_secrets
    tok_path = (token_path or settings.youtube_token_path).expanduser()
    secrets = Path(secrets).expanduser()

    creds = None
    if tok_path.exists():
        creds = Credentials.from_authorized_user_file(str(tok_path), _SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not secrets.exists():
                raise FileNotFoundError(
                    f"YouTube client secrets not found: {secrets}. "
                    "Download from Google Cloud Console → APIs & Services → Credentials."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(secrets), _SCOPES)
            creds = flow.run_local_server(port=0)

        tok_path.parent.mkdir(parents=True, exist_ok=True)
        tok_path.write_text(creds.to_json())

    return build("youtube", "v3", credentials=creds)


# -- Metadata builder -------------------------------------------------------


def build_metadata_from_manifest(
    manifest: MatchManifest,
    privacy: str | None = None,
    title_override: str | None = None,
) -> UploadMetadata:
    """Build YouTube upload metadata from a match manifest."""
    title = title_override or f"{manifest.teams} - {manifest.date} - {manifest.location}"

    total_duration = sum(c.duration for c in manifest.clips)
    clip_count = len(manifest.clips)
    minutes = total_duration / 60

    description = (
        f"Water polo: {manifest.teams}\n"
        f"Date: {manifest.date} {manifest.time}\n"
        f"Location: {manifest.location}\n"
        f"\n"
        f"{clip_count} clip(s), {minutes:.1f} min total"
    )

    tags = ["water polo"]
    for team in manifest.teams.split(" vs "):
        team = team.strip()
        if team and team != "unknown":
            tags.append(team)
    if manifest.date:
        tags.append(manifest.date)
    if manifest.location and manifest.location != "unknown":
        tags.append(manifest.location)

    return UploadMetadata(
        title=title[:100],  # YouTube title limit
        description=description,
        tags=tags,
        privacy=privacy or settings.youtube_default_privacy,
        category_id=settings.youtube_category_id,
    )


# -- Upload ------------------------------------------------------------------


def upload_video(
    service,
    video_path: Path | str,
    metadata: UploadMetadata,
) -> UploadResult:
    """Upload a video to YouTube with resumable upload + exponential backoff.

    Returns an UploadResult with video_id and URL.
    """
    from googleapiclient.http import MediaFileUpload

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    body = {
        "snippet": {
            "title": metadata.title,
            "description": metadata.description,
            "tags": metadata.tags,
            "categoryId": metadata.category_id,
        },
        "status": {
            "privacyStatus": metadata.privacy,
        },
    }

    media = MediaFileUpload(
        str(video_path),
        chunksize=10 * 1024 * 1024,  # 10 MB chunks
        resumable=True,
        mimetype="video/mp4",
    )

    request = service.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    response = _resumable_upload(request)
    video_id = response["id"]
    return UploadResult(
        video_id=video_id,
        url=f"https://youtu.be/{video_id}",
    )


def _resumable_upload(request) -> dict:
    """Execute resumable upload with exponential backoff retry."""
    response = None
    retry = 0
    while response is None:
        try:
            _, response = request.next_chunk()
        except http.client.HTTPException:
            # Transient HTTP error — retry
            if retry >= MAX_RETRIES:
                raise
            retry += 1
            _backoff_sleep(retry)
        except Exception as e:
            # Check for retriable API errors
            if hasattr(e, "resp") and e.resp.status in RETRIABLE_STATUS_CODES:
                if retry >= MAX_RETRIES:
                    raise
                retry += 1
                _backoff_sleep(retry)
            else:
                raise
    return response


def _backoff_sleep(retry: int) -> None:
    """Sleep with exponential backoff + jitter."""
    wait = min(2**retry + random.random(), 60)
    time.sleep(wait)


# -- Convenience wrapper ----------------------------------------------------


def upload_from_manifest(
    video_path: Path | str,
    manifest: MatchManifest,
    privacy: str | None = None,
    title: str | None = None,
    service=None,
) -> UploadResult:
    """One-call upload: build metadata from manifest, authenticate, upload."""
    meta = build_metadata_from_manifest(manifest, privacy=privacy, title_override=title)
    if service is None:
        service = get_authenticated_service()
    return upload_video(service, video_path, meta)


# -- Playlists ---------------------------------------------------------------


def create_playlist(
    service,
    title: str,
    description: str = "",
    privacy: str = "unlisted",
) -> str:
    """Create a YouTube playlist and return its ID."""
    body = {
        "snippet": {
            "title": title,
            "description": description,
        },
        "status": {
            "privacyStatus": privacy,
        },
    }
    response = service.playlists().insert(
        part="snippet,status",
        body=body,
    ).execute()
    return response["id"]


def find_playlist(service, title: str) -> str | None:
    """Find an existing playlist by title. Returns playlist ID or None."""
    response = service.playlists().list(
        part="snippet",
        mine=True,
        maxResults=50,
    ).execute()
    for item in response.get("items", []):
        if item["snippet"]["title"] == title:
            return item["id"]
    return None


def get_or_create_playlist(
    service,
    title: str,
    description: str = "",
    privacy: str = "unlisted",
) -> str:
    """Find an existing playlist by title, or create one. Returns playlist ID."""
    pid = find_playlist(service, title)
    if pid:
        return pid
    return create_playlist(service, title, description, privacy)


def add_video_to_playlist(service, playlist_id: str, video_id: str) -> None:
    """Add a video to a playlist."""
    body = {
        "snippet": {
            "playlistId": playlist_id,
            "resourceId": {
                "kind": "youtube#video",
                "videoId": video_id,
            },
        },
    }
    service.playlistItems().insert(
        part="snippet",
        body=body,
    ).execute()
