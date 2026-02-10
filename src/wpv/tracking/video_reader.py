"""Video reader: persistent prefetch thread + cv2 seeks."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


class VideoReader:
    """Video reader with a persistent decode thread and separate seek path.

    A single background thread owns a cv2.VideoCapture and reads frames
    sequentially into a bounded buffer.  The tracker (main thread) consumes
    frames from the buffer, so decode overlaps with HSV + CNN processing.

    When the tracker needs random seeks (SEARCH_FORWARD / REWIND_BACKWARD),
    those happen on the calling thread with an independent VideoCapture so
    the prefetch thread is never interrupted.

    Calling ``sequential_frames()`` again (e.g. after a search/resume)
    simply increments a generation counter.  The producer abandons the old
    read and starts from the new position; the consumer skips stale frames
    from previous generations.
    """

    _BUFFER = 8

    def __init__(self, video_path: str | Path):
        self._path = Path(video_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Video not found: {self._path}")

        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self._path}")
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Prefetch machinery
        self._frame_q: queue.Queue[tuple[int, int, np.ndarray | None]] = queue.Queue(
            maxsize=self._BUFFER,
        )
        self._cmd_q: queue.Queue[tuple[int, int, int]] = queue.Queue()
        self._shutdown = threading.Event()
        self._generation = 0

        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    # -- properties ----------------------------------------------------------

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

    # -- persistent reader thread --------------------------------------------

    def _reader_loop(self) -> None:
        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            log.error("Prefetch thread: cannot open %s", self._path)
            return

        try:
            while not self._shutdown.is_set():
                # Wait for a read command: (generation, start, end)
                try:
                    gen, start, end = self._cmd_q.get(timeout=1)
                except queue.Empty:
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, start)

                for idx in range(start, end):
                    if self._shutdown.is_set():
                        break
                    # A newer command means our read is obsolete
                    if not self._cmd_q.empty():
                        break

                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Put frame, but keep checking for abort / shutdown
                    put_ok = False
                    while not self._shutdown.is_set() and not put_ok:
                        if not self._cmd_q.empty():
                            break
                        try:
                            self._frame_q.put((gen, idx, frame), timeout=0.5)
                            put_ok = True
                        except queue.Full:
                            pass
                    if not put_ok:
                        break

                # End-of-read sentinel (skip if aborted by newer command)
                if not self._shutdown.is_set() and self._cmd_q.empty():
                    try:
                        self._frame_q.put((gen, -1, None), timeout=2)
                    except queue.Full:
                        pass
        finally:
            cap.release()

    # -- sequential reads (consumer side) ------------------------------------

    def sequential_frames(
        self, start: int = 0, end: int | None = None
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (frame_index, bgr_array) with background prefetch.

        Safe to call multiple times (e.g. after a search/resume); the
        producer resets automatically via the generation counter.
        """
        if end is None:
            end = self._frame_count

        self._generation += 1
        gen = self._generation

        # Send read command (implicitly obsoletes any in-flight read)
        self._cmd_q.put((gen, start, end))

        while True:
            try:
                item_gen, idx, frame = self._frame_q.get(timeout=120)
            except queue.Empty:
                break
            if item_gen != gen:
                continue  # stale frame from a previous read
            if idx == -1:
                break  # end sentinel
            yield idx, frame

    # -- random seeks (on calling thread) ------------------------------------

    def seek_frame(self, frame_idx: int) -> np.ndarray | None:
        """Read a single frame by index via cv2 seek."""
        result = self.seek_frames([frame_idx])
        return result.get(frame_idx)

    def seek_frames(self, frame_indices: list[int]) -> dict[int, np.ndarray]:
        """Read specific frames by index using cv2 seek."""
        result: dict[int, np.ndarray] = {}
        cap = cv2.VideoCapture(str(self._path))
        for idx in sorted(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                result[idx] = frame
        cap.release()
        return result

    # -- cleanup -------------------------------------------------------------

    def close(self) -> None:
        """Shut down the prefetch thread."""
        self._shutdown.set()
        self._thread.join(timeout=5)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
