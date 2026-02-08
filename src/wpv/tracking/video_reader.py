"""Dual-mode video reader: sequential cv2 + NVDEC seek for H.265 content."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


def _nv12_to_bgr_bt709(nv12: np.ndarray, height: int) -> np.ndarray:
    """Convert NV12 buffer to BGR using BT.709 coefficients.

    OpenCV's COLOR_YUV2BGR_NV12 uses BT.601 which is wrong for HD/4K H.265
    content. BT.709 is the correct standard for HD video.
    """
    # NV12 layout: Y plane (height rows), UV plane (height/2 rows, interleaved U,V)
    y_plane = nv12[:height, :].astype(np.float32)
    uv_plane = nv12[height:, :]
    width = y_plane.shape[1]

    # Separate U and V (interleaved in NV12)
    u = uv_plane[:, 0::2].astype(np.float32) - 128.0
    v = uv_plane[:, 1::2].astype(np.float32) - 128.0

    # Upsample UV to full resolution (nearest neighbor, same as hardware)
    u = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)[:height, :width]
    v = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)[:height, :width]

    # BT.709 YUV -> RGB
    r = y_plane + 1.5748 * v
    g = y_plane - 0.1873 * u - 0.4681 * v
    b = y_plane + 1.8556 * u

    bgr = np.stack([b, g, r], axis=-1)
    return np.clip(bgr, 0, 255).astype(np.uint8)


class VideoReader:
    """Dual-mode video reader: sequential cv2 reads + NVDEC random seeks.

    Sequential mode uses cv2.VideoCapture for efficient forward iteration.
    Seek mode uses PyNvVideoCodec (NVDEC) for fast random access on H.265,
    falling back to cv2 seek if NVDEC is unavailable.
    """

    def __init__(self, video_path: str | Path):
        self._path = Path(video_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Video not found: {self._path}")

        # Probe video metadata via cv2
        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self._path}")
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def sequential_frames(
        self, start: int = 0, end: int | None = None
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (frame_index, bgr_array) sequentially from start to end.

        Uses cv2.VideoCapture for efficient sequential reads.
        """
        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self._path}")

        if end is None:
            end = self._frame_count

        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        try:
            for idx in range(start, end):
                ret, frame = cap.read()
                if not ret:
                    break
                yield idx, frame
        finally:
            cap.release()

    def seek_frame(self, frame_idx: int) -> np.ndarray | None:
        """Read a single frame by index. Uses NVDEC if available."""
        result = self.seek_frames([frame_idx])
        return result.get(frame_idx)

    def seek_frames(self, frame_indices: list[int]) -> dict[int, np.ndarray]:
        """Read specific frames by index. Uses NVDEC if available, else cv2.

        Returns dict mapping frame_index -> BGR numpy array.
        """
        try:
            return self._seek_nvdec(frame_indices)
        except (ImportError, Exception):
            return self._seek_cv2(frame_indices)

    def _seek_nvdec(self, frame_indices: list[int]) -> dict[int, np.ndarray]:
        """Read frames using NVIDIA hardware decoder."""
        import torch  # noqa: F401 — needed for DLPack interop
        import PyNvVideoCodec as nvc

        mp4_str = str(self._path)
        result: dict[int, np.ndarray] = {}

        for target_frame in sorted(frame_indices):
            try:
                demuxer = nvc.CreateDemuxer(mp4_str)
                dec = nvc.CreateDecoder(gpuid=0, codec=demuxer.GetNvCodecId())
                ts = demuxer.TimestampFromFrame(target_frame)
                seek_packet = demuxer.Seek(ts)

                got_frame = False
                frames = dec.Decode(seek_packet)
                for f in frames:
                    nv12 = torch.from_dlpack(f).numpy().copy()
                    bgr = _nv12_to_bgr_bt709(nv12, self._height)
                    result[target_frame] = bgr
                    got_frame = True
                    break

                if not got_frame:
                    for i, packet in enumerate(demuxer):
                        frames = dec.Decode(packet)
                        for f in frames:
                            nv12 = torch.from_dlpack(f).numpy().copy()
                            bgr = _nv12_to_bgr_bt709(nv12, self._height)
                            result[target_frame] = bgr
                            got_frame = True
                            break
                        if got_frame or i > 100:
                            break
            except Exception:
                # Fall back to cv2 for this frame
                cap = cv2.VideoCapture(mp4_str)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, frame = cap.read()
                if ret:
                    result[target_frame] = frame
                cap.release()

        return result

    def _seek_cv2(self, frame_indices: list[int]) -> dict[int, np.ndarray]:
        """Fallback: read frames using cv2 seek."""
        result: dict[int, np.ndarray] = {}
        cap = cv2.VideoCapture(str(self._path))
        for idx in sorted(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                result[idx] = frame
        cap.release()
        return result
