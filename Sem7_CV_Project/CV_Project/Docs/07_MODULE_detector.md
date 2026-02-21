# 07 — Module: `detector.py`

**Syllabus Topic:** Object Detection
**File:** `parking_counter/detector.py` (180 lines)

---

## Purpose

Wraps YOLOv8 nano (via the `ultralytics` library) to:
1. Lazily load the model weights on first inference call
2. Run inference and filter to vehicle COCO classes only
3. Return structured `Detection` dataclass objects
4. Draw bounding boxes for visualisation

---

## Dataclass: `Detection`

```python
@dataclass
class Detection:
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2) pixel coordinates
    confidence: float                  # YOLO confidence score (0.0–1.0)
    class_id: int                      # COCO class ID
    label: str                         # Human-readable label ("car", "truck", etc.)
```

### COCO Vehicle Class IDs Used

| `class_id` | `label` |
|-----------|--------|
| 2 | `"car"` |
| 3 | `"motorcycle"` |
| 5 | `"bus"` |
| 7 | `"truck"` |

These are defined in `VehicleDetector.COCO_VEHICLE_NAMES`:
```python
COCO_VEHICLE_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
```

---

## Class: `VehicleDetector`

### Constructor

```python
VehicleDetector(
    model_path: str | Path = "assets/yolov8n.pt",
    confidence: float = 0.45,
    iou_threshold: float = 0.45,
    vehicle_classes: List[int] | None = None,  # defaults to [2, 3, 5, 7]
    device: str = "cpu",
)
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `model_path` | `"assets/yolov8n.pt"` | Relative to CWD (`parking_counter/`). Auto-downloaded if missing. |
| `confidence` | 0.45 | Min confidence. Detections below this are discarded before NMS. |
| `iou_threshold` | 0.45 | NMS IoU threshold. |
| `vehicle_classes` | [2,3,5,7] | If `None` → defaults to `[2, 3, 5, 7]` |
| `device` | `"cpu"` | `"cpu"` | `"cuda"` | `"mps"`. ALWAYS use `"cpu"` on Windows without verified CUDA. |

**Internal state:** `self._model = None` — model not loaded until first `detect()` call.

---

### Method: `load() → None`

Explicitly loads the YOLO model.

```python
def load(self) -> None:
    from ultralytics import YOLO
    self._model = YOLO(self.model_path)
    # Warm-up: run one dummy inference to initialise layers
    dummy = np.zeros((1, 640, 640, 3), dtype=np.uint8)
    self._model.predict(dummy, verbose=False, device=self.device)
```

- The `from ultralytics import YOLO` is **inside** `load()` to defer the heavy import.
- Warm-up ensures the first real inference doesn't have initialisation latency.
- Called automatically by `detect()` on first call, but can be called explicitly to pre-load.

---

### Method: `detect(frame: np.ndarray) → List[Detection]`

Main inference method.

```python
def detect(self, frame: np.ndarray) -> List[Detection]:
    if self._model is None:
        self.load()

    try:
        results = self._model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=self.vehicle_classes,    # YOLO filters classes internally
            verbose=False,
            device=self.device,
        )
    except Exception as exc:
        print(f"[Detector] Inference error: {exc}")
        return []   # Graceful degradation

    detections: List[Detection] = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                class_id=cls_id,
                label=self.COCO_VEHICLE_NAMES.get(cls_id, str(cls_id)),
            ))
    return detections
```

**Key points:**
- `classes=self.vehicle_classes` is passed to `model.predict()` — YOLO filters at inference time, not post-hoc
- `box.xyxy[0]` is a tensor; values are converted to Python `int` via `int(v)`
- Any exception returns `[]` (empty list) rather than crashing
- `verbose=False` suppresses YOLO's default logging output

---

### Method: `draw_detections(frame, detections, color=(255,165,0), thickness=2) → np.ndarray`

Draws bounding boxes on a **copy** of `frame`.

```python
for det in detections:
    x1, y1, x2, y2 = det.bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    label = f"{det.label} {det.confidence:.2f}"
    cv2.putText(out, label, (x1, max(y1-6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
```

Default color is orange `(255, 165, 0)` BGR.

**Note:** In `pipeline.py`, the overlay drawing is done inline (not via this method)
using `(0, 165, 255)` orange color and thinner `thickness=1`.

---

## Imports

```python
from dataclasses import dataclass
from pathlib import Path
from typing import List
import numpy as np
# cv2 imported at module bottom to avoid circular issue with PyQt5:
import cv2  # noqa: E402
```

Note the unusual `import cv2` at the **end** of the file. This is intentional to avoid
import ordering issues. It is fine because `cv2` is only used in `draw_detections()`.

---

## How `pipeline.py` Uses This Module

```python
# In PipelineThread._process_frame():
try:
    detections: List[Detection] = self._detector.detect(frame)
    self._last_detections = detections
    self._detector_failed = False
except Exception as exc:
    if not self._detector_failed:
        self.error_occurred.emit(f"Detection error: {exc}")
    self._detector_failed = True
    detections = self._last_detections  # reuse cached
```

**Note:** `_detector_failed` is set to `True` on exception. When `True`, the pipeline
auto-switches from overlap method to pixel_count fallback.

---

## YOLOv8 Nano Details

- Model: `yolov8n` (nano variant — smallest, fastest)
- Input: any size (YOLO auto-pads/letterboxes internally)
- Training data: COCO (80 classes)
- Inference time on CPU: ~50–200ms depending on image size and hardware
- Weights file: ~6MB download
- Auto-download: `ultralytics` fetches from GitHub if `assets/yolov8n.pt` is missing
