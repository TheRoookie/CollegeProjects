"""
detector.py
-----------
Vehicle detection using YOLOv8 nano via the Ultralytics library.

The detector wraps the YOLO model and filters detections to vehicle
classes only (car, motorcycle, bus, truck in the COCO dataset).

Syllabus topic covered: **Object Detection**
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class Detection:
    """A single vehicle detection result."""

    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2) in pixel coords
    confidence: float
    class_id: int
    label: str


class VehicleDetector:
    """
    Detects vehicles in a frame using YOLOv8 nano.

    Parameters
    ----------
    model_path : str | Path
        Path to the ``yolov8n.pt`` weights file.
        If the file doesn't exist the Ultralytics library will download it.
    confidence : float
        Minimum confidence score to keep a detection.
    iou_threshold : float
        IoU threshold for NMS.
    vehicle_classes : list[int]
        COCO class IDs considered as vehicles.
        Defaults: car=2, motorcycle=3, bus=5, truck=7.
    device : str
        Inference device: ``"cpu"``, ``"cuda"``, or ``"mps"``.
    """

    # COCO class names for reference
    COCO_VEHICLE_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def __init__(
        self,
        model_path: str | Path = "assets/yolov8n.pt",
        confidence: float = 0.45,
        iou_threshold: float = 0.45,
        vehicle_classes: List[int] | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_path = str(model_path)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.vehicle_classes = vehicle_classes or [2, 3, 5, 7]
        self.device = device

        self._model = None  # Lazy-loaded on first call to detect()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Explicitly load / download the YOLO model weights."""
        from ultralytics import YOLO  # type: ignore[import]
        self._model = YOLO(self.model_path)
        # Warm-up inference to initialise layers
        dummy = np.zeros((1, 640, 640, 3), dtype=np.uint8)
        self._model.predict(dummy, verbose=False, device=self.device)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on *frame* and return a list of vehicle detections.

        The model is loaded lazily on the first call.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (as returned by OpenCV).

        Returns
        -------
        list[Detection]
            All vehicle detections above the confidence threshold.
        """
        if self._model is None:
            self.load()

        try:
            results = self._model.predict(
                frame,
                conf=self.confidence,
                iou=self.iou_threshold,
                classes=self.vehicle_classes,
                verbose=False,
                device=self.device,
            )
        except Exception as exc:
            # Gracefully degrade: return empty list if inference fails
            print(f"[Detector] Inference error: {exc}")
            return []

        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=cls_id,
                        label=self.COCO_VEHICLE_NAMES.get(cls_id, str(cls_id)),
                    )
                )
        return detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        color: tuple[int, int, int] = (255, 165, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw bounding boxes for *detections* onto a copy of *frame*.

        Returns the annotated frame (does not modify in-place).
        """
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            label = f"{det.label} {det.confidence:.2f}"
            cv2.putText(
                out, label, (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )
        return out


# Import guard so cv2 is available when draw_detections is called
import cv2  # noqa: E402
