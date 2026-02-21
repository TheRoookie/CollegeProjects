# 10 — Module: `pipeline.py`

**File:** `parking_counter/pipeline.py` (277 lines)

---

## Purpose

The central orchestration module. Runs in its own `QThread` so it never blocks the GUI.

**Data flow per frame:**
```
VideoHandler → VehicleDetector → CentroidTracker → per-spot classification → draw overlays
                                                ↑
                                     ROIManager (spot polygons)
```

Emits PyQt5 signals that the GUI connects to for live display and statistics.

---

## Class: `PipelineThread(QThread)`

### Signals

```python
frame_ready    = pyqtSignal(np.ndarray)   # Emitted each frame; arg = annotated BGR frame
stats_updated  = pyqtSignal(int, int, int) # Emitted each frame; args = (total, empty, occupied)
error_occurred = pyqtSignal(str)           # Emitted on non-fatal errors; arg = message string
finished_signal = pyqtSignal()             # Emitted when stream ends or stop() called
```

### Constructor

```python
PipelineThread(
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
)
```

| Parameter | Description |
|-----------|-------------|
| `video_handler` | `VideoHandler` instance (not yet opened) |
| `roi_manager` | `ROIManager` with loaded spots |
| `detector` | `VehicleDetector` (model not yet loaded) |
| `tracker` | `CentroidTracker` (fresh) |
| `classifier` | `SpaceClassifier` with pixel_count backend (used as YOLO fallback) |
| `display_cfg` | Dict from `config.yaml → display` section |
| `detection_fps` | How many times per second to run full YOLO detection |
| `status_debounce` | Passed to `ParkingSpot.update_status(debounce=...)` |
| `use_detection_overlap` | If `True`, use YOLO bbox overlap; if `False`, use pixel_count |
| `overlap_threshold` | Fraction of spot polygon that must be covered to count as occupied |

**Key internal state:**
```python
self._running: bool = False
self._last_detections: List[Detection] = []         # Cache for non-detection frames
self._detector_failed: bool = False                 # Switches to pixel_count fallback
```

---

### Method: `run() → None`

Called automatically by `QThread.start()`. Main processing loop.

```python
def run(self) -> None:
    self._running = True

    try:
        self._video.open()
    except IOError as exc:
        self.error_occurred.emit(str(exc))
        self.finished_signal.emit()
        return

    # Compute detection frequency
    video_fps: float = self._video.fps or 30.0
    if self._detection_fps > 0:
        detect_every = max(1, round(video_fps / self._detection_fps))
    else:
        detect_every = 1  # Every frame

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
```

**`detect_every` formula:**
```
video_fps=30, detection_fps=1  → detect_every = max(1, round(30/1)) = 30
video_fps=30, detection_fps=5  → detect_every = max(1, round(30/5)) = 6
video_fps=30, detection_fps=30 → detect_every = 1 (every frame)
detection_fps=0                → detect_every = 1 (every frame, legacy mode)
```

### Method: `stop() → None`

```python
def stop(self) -> None:
    self._running = False
```

Sets flag; the `for` loop in `run()` breaks on the next iteration.

---

### Method: `_process_frame(frame, run_detection=True) → np.ndarray`

The per-frame CV pipeline.

#### When `run_detection=True`:

**Step 1 — Object Detection:**
```python
try:
    detections: List[Detection] = self._detector.detect(frame)
    self._last_detections = detections
    self._detector_failed = False               # mark recovered
except Exception as exc:
    if not self._detector_failed:               # log only once
        self.error_occurred.emit(f"Detection error: {exc}")
    self._detector_failed = True
    detections = self._last_detections          # use cached
```

**Step 2 — Object Tracking:**
```python
bboxes = [d.bbox for d in detections]
self._tracker.update(bboxes)
```

**Step 3 & 4 — Segmentation + Classification per spot:**
```python
use_overlap_now = self._use_overlap and not self._detector_failed

for spot in self._roi.spots:
    if use_overlap_now:
        # Overlap method: check if any detection covers enough of this spot
        occupied = any(
            _detection_overlap(spot, det.bbox, frame_h, frame_w) >= self._overlap_threshold
            for det in detections
        )
        new_status = "occupied" if occupied else "empty"
    else:
        # Pixel-count fallback
        patch = self._roi.extract_patch(frame, spot)
        raw = self._classifier.classify(patch)
        new_status = "occupied" if raw == SpaceStatus.OCCUPIED else "empty"

    spot.update_status(new_status, self._debounce)
```

#### When `run_detection=False` (non-detection frame):
```python
detections = self._last_detections  # reuse cached; spot statuses unchanged
```
This allows the video to display at full speed without re-running YOLO.

**Step 5 — Draw overlays:**
```python
annotated = self._roi.draw_overlays(
    frame,
    empty_color=tuple(cfg.get("empty_color", [0, 200, 0])),
    occupied_color=tuple(cfg.get("occupied_color", [0, 0, 220])),
    unknown_color=tuple(cfg.get("unknown_color", [200, 200, 0])),
    thickness=cfg.get("box_thickness", 2),
    font_scale=cfg.get("font_scale", 0.5),
)

# Draw vehicle bounding boxes (orange, thin)
for det in detections:
    x1, y1, x2, y2 = det.bbox
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 1)

# Draw tracker trails (yellow)
for track in self._tracker.tracks.values():
    if len(track.history) > 1:
        pts = np.array(track.history, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(annotated, [pts], False, (255, 255, 0), 1)

return annotated
```

---

## Module-Level Function: `_detection_overlap`

```python
def _detection_overlap(
    spot: "ParkingSpot",
    bbox: Tuple[int, int, int, int],
    frame_h: int,
    frame_w: int,
) -> float:
```

**Returns** fraction of `spot`'s polygon area covered by `bbox`, in `[0.0, 1.0]`.

**Full implementation:**
```python
x1, y1, x2, y2 = bbox

# Create binary mask for spot polygon
spot_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
cv2.fillPoly(spot_mask, [spot.polygon], 255)

# Create binary mask for detection bounding box
det_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
cv2.rectangle(
    det_mask,
    (max(0, x1), max(0, y1)),
    (min(frame_w-1, x2), min(frame_h-1, y2)),
    255, -1
)

spot_area = int(np.count_nonzero(spot_mask))
if spot_area == 0:
    return 0.0

intersection = int(np.count_nonzero(cv2.bitwise_and(spot_mask, det_mask)))
return intersection / spot_area
```

This approach:
- Respects the exact polygon shape (not just bounding rectangle)
- Handles partial overlaps correctly
- Returns 0 if the spot polygon has no area
- Clamps bbox coordinates to frame boundaries

**Math:** `overlap_ratio = |spot ∩ detection| / |spot|`

A spot is occupied if `overlap_ratio ≥ overlap_threshold` for ANY detection.

---

## `_detector_failed` Flag Logic

```
Initial state: _detector_failed = False

YOLO succeeds: _detector_failed = False
    → use_overlap_now = True  (overlap method active)

YOLO throws exception:
    First time: emit error_occurred signal (one log message)
    _detector_failed = True
    → use_overlap_now = False (pixel_count fallback active)

YOLO succeeds again: _detector_failed = False (automatically recovered)
```

This ensures:
1. Single error message (not one per frame)
2. Graceful fallback to pixel_count when YOLO is unavailable
3. Automatic recovery when YOLO works again

---

## Imports

```python
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
```
