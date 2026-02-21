"""
video_handler.py
----------------
Handles opening and reading frames from a video source (file or RTSP stream).
Applies per-frame resizing so the rest of the pipeline works on a consistent
resolution regardless of the original video dimensions.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np


class VideoHandler:
    """
    Wraps a cv2.VideoCapture instance and yields preprocessed frames.

    Parameters
    ----------
    source : str | int
        Path to a video file, an RTSP URL, or an integer webcam index.
    process_width : int
        Target width for resizing. Height is scaled to maintain aspect ratio.
    max_fps : int
        Maximum frames per second to process. ``0`` means no cap.
    """

    def __init__(
        self,
        source: str | int,
        process_width: int = 640,
        max_fps: int = 30,
    ) -> None:
        self.source = source
        self.process_width = process_width
        self.max_fps = max_fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_delay: float = 1.0 / max_fps if max_fps > 0 else 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the video capture device / file.  Raises ``IOError`` on failure."""
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video source: {self.source!r}")

    def release(self) -> None:
        """Release the underlying capture resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        """Native FPS reported by the video source (0 when not opened)."""
        if self._cap is None:
            return 0.0
        return float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)

    @property
    def total_frames(self) -> int:
        """Total frame count for file sources. Returns 0 for live streams."""
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def original_size(self) -> Tuple[int, int]:
        """(width, height) of the raw video source."""
        if self._cap is None:
            return (0, 0)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def frames(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yield ``(original_frame, processed_frame)`` tuples.

        * ``original_frame`` – raw BGR frame straight from the capture.
        * ``processed_frame`` – BGR frame resized to ``process_width``.

        Stops automatically when the source ends or ``release()`` is called.
        """
        if self._cap is None:
            raise RuntimeError("Call open() before iterating frames.")

        last_time = time.time()

        while True:
            ret, frame = self._cap.read()
            if not ret:
                # End of file or stream error – loop file for demo purposes.
                if isinstance(self.source, (str, Path)) and Path(str(self.source)).is_file():
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self._cap.read()
                    if not ret:
                        break
                else:
                    break

            processed = self._resize(frame)

            # FPS cap ---------------------------------------------------
            if self._frame_delay > 0:
                elapsed = time.time() - last_time
                sleep_for = self._frame_delay - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
            last_time = time.time()

            yield frame, processed

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        """Resize *frame* so that its width equals ``self.process_width``."""
        h, w = frame.shape[:2]
        if w == self.process_width:
            return frame
        scale = self.process_width / w
        new_h = int(h * scale)
        return cv2.resize(frame, (self.process_width, new_h), interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "VideoHandler":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
