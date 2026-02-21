"""
pipeline.py
-----------
The main computer vision processing pipeline.

Ties together:
  VideoHandler  →  VehicleDetector  →  CentroidTracker
                →  ROIManager       →  SpaceClassifier

Occupancy Classification strategy
----------------------------------
Default (``use_detection_overlap=True``):
  After YOLO detects vehicles, compute what fraction of each parking-spot
  polygon is covered by each detection bounding box.  A spot is marked
  occupied when any detection covers at least ``overlap_threshold`` of
  the spot's polygon area.  This reuses the already-running YOLO results
  and is robust to texture, lighting, and perspective changes.

Fallback (``use_detection_overlap=False``):
  The legacy pixel-count / CNN ``SpaceClassifier`` is used instead.

Runs in its own QThread to keep the PyQt5 UI responsive.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from video_handler import VideoHandler
from roi_manager import ROIManager, ParkingSpot
from detector import VehicleDetector, Detection
from tracker import CentroidTracker
from classifier import SpaceClassifier, SpaceStatus


class PipelineSignals:
    """Mixin of PyQt signals — declared separately so QThread can inherit cleanly."""


class PipelineThread(QThread):
    """
    Worker thread that streams processed frames and parking statistics.

    Signals
    -------
    frame_ready(np.ndarray)
        Emitted for each processed frame (BGR).
    stats_updated(int, int, int)
        Emitted with (total_spots, empty_count, occupied_count).
    error_occurred(str)
        Emitted when a non-fatal error is encountered.
    finished_signal()
        Emitted when the video stream ends or ``stop()`` is called.
    """

    frame_ready = pyqtSignal(np.ndarray)
    stats_updated = pyqtSignal(int, int, int)
    error_occurred = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(
        self,
        video_handler: VideoHandler,
        roi_manager: ROIManager,
        detector: VehicleDetector,
        tracker: CentroidTracker,
        classifier: SpaceClassifier,
        display_cfg: dict,
        detection_fps: float = 1.0,
        status_debounce: int = 3,
        use_detection_overlap: bool = True,
        overlap_threshold: float = 0.25,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._video = video_handler
        self._roi = roi_manager
        self._detector = detector
        self._tracker = tracker
        self._classifier = classifier
        self._display_cfg = display_cfg
        self._detection_fps = detection_fps
        self._debounce = status_debounce
        self._use_overlap = use_detection_overlap
        self._overlap_threshold = overlap_threshold
        self._running = False
        # Cache last detections so non-detection frames still show bounding boxes
        self._last_detections: List[Detection] = []
        # Track whether the YOLO detector is currently in a failed state.
        # When True the system automatically falls back to pixel_count classification.
        self._detector_failed: bool = False

    # ------------------------------------------------------------------
    # Thread lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main loop — called automatically by QThread.start()."""
        self._running = True
        try:
            self._video.open()
        except IOError as exc:
            self.error_occurred.emit(str(exc))
            self.finished_signal.emit()
            return

        # Compute how many frames to skip between each detection pass.
        # video_fps=30, detection_fps=1  →  detect_every=30  (1 detection per second)
        # video_fps=30, detection_fps=5  →  detect_every=6
        # detection_fps=0 means "detect on every frame" (legacy behaviour)
        video_fps: float = self._video.fps or 30.0
        if self._detection_fps > 0:
            detect_every = max(1, round(video_fps / self._detection_fps))
        else:
            detect_every = 1

        frame_idx = 0
        for _raw, frame in self._video.frames():
            if not self._running:
                break

            run_detection = (frame_idx % detect_every == 0)
            annotated = self._process_frame(frame, run_detection=run_detection)
            self.frame_ready.emit(annotated)
            self.stats_updated.emit(
                self._roi.total,
                self._roi.empty_count,
                self._roi.occupied_count,
            )
            frame_idx += 1

        self._video.release()
        self.finished_signal.emit()

    def stop(self) -> None:
        """Request the thread to stop after the current frame."""
        self._running = False

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray, run_detection: bool = True) -> np.ndarray:
        """
        Run the CV pipeline on a single frame.

        Parameters
        ----------
        run_detection : bool
            When True, run the full detection + classification pipeline and
            update spot statuses.  When False, skip heavy inference and reuse
            the statuses from the previous detection frame — the video still
            displays at full speed with the last-known overlay.

        Steps (when run_detection=True)
        --------------------------------
        1. Object Detection  — find vehicles with YOLOv8.
        2. Object Tracking   — update centroid tracker.
        3. Segmentation      — extract per-spot ROI patches.
        4. Classification    — label each spot empty / occupied.
        5. Overlay drawing   — annotate frame for display.
        """

        if run_detection:
            # ---- 1. Object Detection --------------------------------
            try:
                detections: List[Detection] = self._detector.detect(frame)
                self._last_detections = detections
                self._detector_failed = False          # recovered
            except Exception as exc:
                if not self._detector_failed:          # log only on first failure
                    self.error_occurred.emit(f"Detection error: {exc}")
                self._detector_failed = True
                detections = self._last_detections

            # ---- 2. Object Tracking ---------------------------------
            bboxes = [d.bbox for d in detections]
            self._tracker.update(bboxes)

            # ---- 3 & 4. Segmentation + Classification per spot ------
            frame_h, frame_w = frame.shape[:2]

            # Decide which classification mode to use this frame:
            #   - overlap     : YOLO is working correctly  → use bbox overlap
            #   - pixel_count : YOLO errored → adaptive-threshold classifier
            use_overlap_now = self._use_overlap and not self._detector_failed

            for spot in self._roi.spots:
                if use_overlap_now:
                    # Overlap method: accurate on any texture / lighting
                    occupied = any(
                        _detection_overlap(spot, det.bbox, frame_h, frame_w)
                        >= self._overlap_threshold
                        for det in detections
                    )
                    new_status = "occupied" if occupied else "empty"
                else:
                    # ---- Classical CV fallback (Image Segmentation + Classification)
                    # ----------------------------------------------------------------
                    # Pipeline: grayscale → small Gaussian blur → adaptive
                    # threshold (BINARY_INV, block=25, C=16) → dilation →
                    # non-zero pixel count.
                    #
                    # Key insight for cobblestone/brick lots:
                    #   Brick texture → HIGH pixel count (each tile darker
                    #   than mortar border → many locally-dark pixels survive)
                    #   Car body → LOW pixel count (large smooth region →
                    #   few pixels deviate from local mean)
                    # Therefore: count < threshold → OCCUPIED.
                    patch = self._roi.extract_patch(frame, spot)
                    raw = self._classifier.classify(patch)
                    new_status = "occupied" if raw == SpaceStatus.OCCUPIED else "empty"

                spot.update_status(new_status, self._debounce)
        else:
            # Non-detection frame: keep existing spot statuses and cached detections
            detections = self._last_detections

        # ---- 5. Draw overlays ----------------------------------------
        cfg = self._display_cfg
        annotated = self._roi.draw_overlays(
            frame,
            empty_color=tuple(cfg.get("empty_color", [0, 200, 0])),
            occupied_color=tuple(cfg.get("occupied_color", [0, 0, 220])),
            unknown_color=tuple(cfg.get("unknown_color", [200, 200, 0])),
            thickness=cfg.get("box_thickness", 2),
            font_scale=cfg.get("font_scale", 0.5),
        )

        # Draw vehicle bounding boxes (optional, small orange boxes)
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 1)

        # Draw tracker trails
        for track in self._tracker.tracks.values():
            if len(track.history) > 1:
                pts = np.array(track.history, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(annotated, [pts], False, (255, 255, 0), 1)

        return annotated


# ---------------------------------------------------------------------------
# Overlap helper (module-level so it can be unit-tested independently)
# ---------------------------------------------------------------------------

def _detection_overlap(
    spot: "ParkingSpot",
    bbox: Tuple[int, int, int, int],
    frame_h: int,
    frame_w: int,
) -> float:
    """
    Return the fraction of *spot*'s polygon area that is covered by *bbox*.

    Uses binary masks so the polygon shape is respected exactly.
    Returns a value in [0.0, 1.0].
    """
    x1, y1, x2, y2 = bbox

    # Spot polygon mask
    spot_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.fillPoly(spot_mask, [spot.polygon], 255)

    # Detection bbox mask
    det_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.rectangle(
        det_mask,
        (max(0, x1), max(0, y1)),
        (min(frame_w - 1, x2), min(frame_h - 1, y2)),
        255,
        -1,
    )

    spot_area = int(np.count_nonzero(spot_mask))
    if spot_area == 0:
        return 0.0

    intersection = int(np.count_nonzero(cv2.bitwise_and(spot_mask, det_mask)))
    return intersection / spot_area


