# 11 — Module: `ui.py`

**File:** `parking_counter/ui.py` (475 lines)

---

## Purpose

PyQt5 main window containing:
- A toolbar with 4 action buttons
- A live video display widget
- An analytics sidebar with real-time counts and event log
- Full pipeline lifecycle management (start/stop)
- Connection to `PipelineThread` via Qt signals/slots

---

## Class: `VideoLabel(QLabel)`

A `QLabel` subclass that displays video frames while preserving aspect ratio.

### Key Behaviour

```python
class VideoLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 270)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pixmap: Optional[QPixmap] = None
        self.setText("No video loaded")
        # Placeholder styling: dark background, grey text
        self.setStyleSheet(
            "QLabel { background-color: #1a1a2e; color: #555; border-radius: 8px; }"
        )
```

### Method: `set_frame(frame: np.ndarray) → None`

Converts a BGR numpy frame to a scaled `QPixmap`:
```python
def set_frame(self, frame: np.ndarray) -> None:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    self._pixmap = QPixmap.fromImage(qimg)
    self._refresh()
```

### Method: `_refresh() → None`

Called both from `set_frame()` and `resizeEvent()`:
```python
def _refresh(self) -> None:
    if self._pixmap:
        scaled = self._pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(scaled)
```

`Qt.KeepAspectRatio` — maintains 16:9 (or whatever the video's ratio is).
`Qt.SmoothTransformation` — bilinear downscaling for clean display.

---

## Class: `AnalyticsSidebar(QWidget)`

Fixed-width (220px) right panel showing parking statistics.

### Widgets

| Widget | Style | Content |
|--------|-------|---------|
| `_lbl_total` | `color: #e0e0e0` | "Total Spaces: N" |
| `_lbl_empty` | `color: #00c853` (green) | "Available: N" |
| `_lbl_occupied` | `color: #ff1744` (red) | "Occupied: N" |
| `_bar` (QProgressBar) | red chunk, dark bg | Occupancy % (0–100) |
| `_log` (QTextEdit) | readonly, dark bg, tiny font | Event log messages |

### Method: `update_stats(total, empty, occupied) → None`

```python
def update_stats(self, total: int, empty: int, occupied: int) -> None:
    self._lbl_total.setText(f"Total Spaces:  {total}")
    self._lbl_empty.setText(f"Available:     {empty}")
    self._lbl_occupied.setText(f"Occupied:      {occupied}")
    pct = int(occupied / total * 100) if total > 0 else 0
    self._bar.setValue(pct)
    self._bar.setFormat(f"{pct}%")
```

### Method: `log(msg: str) → None`

Appends a message to the event log text area.

---

## Class: `MainWindow(QMainWindow)`

### Constructor

```python
MainWindow(config_path: str = "config.yaml")
```

1. Reads `config.yaml` via `yaml.safe_load` into `self._cfg`
2. Sets `self._pipeline_thread = None` and `self._video_source = None`
3. Calls `_build_ui()` and `_apply_dark_theme()`

### Layout Structure

```
QMainWindow
└─ QToolBar (top)
│    ├─ QPushButton "Open Video"  → _on_open_video()
│    ├─ QPushButton "Load ROIs"   → _on_load_rois()
│    ├─ QPushButton "Draw ROIs"   → _on_draw_rois()
│    ├─ [separator]
│    └─ QPushButton "▶ Start"     → _on_start_stop()
└─ QWidget (central)
│    └─ QHBoxLayout
│         ├─ VideoLabel (stretch=4)
│         ├─ QFrame (VLine separator)
│         └─ AnalyticsSidebar (stretch=0, fixed width=220)
└─ QStatusBar (bottom)
     └─ showMessage(...)
```

### Colour Scheme

| Element | Hex Color |
|---------|-----------|
| Main window background | `#12121f` |
| Toolbar background | `#1e1e3a` |
| Status bar background | `#1e1e3a` |
| Toolbar buttons (default) | `#2a2a4a` (hover: `#3a3a6a`) |
| Start button (running) | `#00897b` / teal green |
| Stop button (active) | `#c62828` / dark red |

---

## Button Action Methods

### `_on_open_video()`
```python
path, _ = QFileDialog.getOpenFileName(
    self, "Open Video File", "",
    "Video Files (*.mp4 *.avi *.mkv *.mov *.webm);;All Files (*)"
)
if path:
    self._video_source = path
    # updates status bar and sidebar log
```

### `_on_load_rois()`
1. Opens file dialog filtered to `*.json`
2. Calls `roi_manager.load()` on selected file
3. Shows success/failure in status bar

### `_on_draw_rois()`
```python
subprocess.Popen(
    [
        sys.executable, "draw_rois.py",
        "--video", self._video_source,
        "--output", self._cfg["roi"]["coordinates_file"],
        "--config", "config.yaml",
    ],
    cwd=str(Path(__file__).parent)  # run from parking_counter/ directory
)
```
`--config` is passed so `draw_rois.py` reads the same `process_width`.

### `_on_start_stop()`
Toggles between `_start_pipeline()` and `_stop_pipeline()`.

---

## Pipeline Start Logic: `_start_pipeline()`

Constructs all pipeline objects from config and starts the thread:

```python
vcfg  = self._cfg["video"]
det_cfg = self._cfg["detection"]
trk_cfg = self._cfg["tracker"]
cls_cfg = self._cfg["classifier"]
disp_cfg = self._cfg["display"]

video_handler = VideoHandler(
    source=self._video_source,
    process_width=vcfg["process_width"],
    max_fps=vcfg["max_fps"],
)

detector = VehicleDetector(
    model_path=det_cfg["model_weights"],
    confidence=det_cfg["confidence_threshold"],
    iou_threshold=det_cfg["iou_threshold"],
    vehicle_classes=det_cfg["vehicle_classes"],
    device=det_cfg["device"],
)

tracker = CentroidTracker(
    max_distance=trk_cfg["max_distance"],
    max_disappeared=trk_cfg["max_disappeared"],
)

# CRITICAL: classifier selection
method = cls_cfg.get("method", "overlap")
if method == "cnn":
    classifier = SpaceClassifier(method="cnn", checkpoint_path=cls_cfg["cnn_checkpoint"])
else:
    # "overlap" and "pixel_count" both use pixel_count backend
    # (overlap method is handled in pipeline.py, not here)
    classifier = SpaceClassifier(
        method="pixel_count",
        pixel_threshold=cls_cfg.get("pixel_threshold", 0.25),
        blur_kernel=cls_cfg.get("blur_kernel", 5),
        adaptive_block=cls_cfg.get("adaptive_block", 11),
        adaptive_c=cls_cfg.get("adaptive_c", 2),
    )

use_overlap = (method == "overlap")

self._pipeline_thread = PipelineThread(
    video_handler=video_handler,
    roi_manager=roi_mgr,
    detector=detector,
    tracker=tracker,
    classifier=classifier,
    display_cfg=disp_cfg,
    detection_fps=vcfg.get("detection_fps", 1.0),
    status_debounce=cls_cfg.get("status_debounce", 2),
    use_detection_overlap=use_overlap,
    overlap_threshold=cls_cfg.get("overlap_threshold", 0.25),
    parent=self,
)
```

### Signal Connections:
```python
self._pipeline_thread.frame_ready.connect(self._on_frame_ready)
self._pipeline_thread.stats_updated.connect(self._on_stats_updated)
self._pipeline_thread.error_occurred.connect(self._on_pipeline_error)
self._pipeline_thread.finished_signal.connect(self._on_pipeline_finished)

self._pipeline_thread.start()
```

---

## Slot Methods

| Slot | Connected Signal | Action |
|------|-----------------|--------|
| `_on_frame_ready(frame)` | `frame_ready` | Calls `self._video_label.set_frame(frame)` |
| `_on_stats_updated(total,empty,occ)` | `stats_updated` | Calls `self._sidebar.update_stats(...)` |
| `_on_pipeline_error(msg)` | `error_occurred` | Logs to sidebar + status bar |
| `_on_pipeline_finished()` | `finished_signal` | Calls `_stop_pipeline()` |

---

## ROI Manager Caching: `_get_or_create_roi_manager()`

```python
def _get_or_create_roi_manager(self) -> ROIManager:
    if not hasattr(self, "_roi_manager"):
        roi_file = self._cfg["roi"]["coordinates_file"]
        self._roi_manager = ROIManager(roi_file)
        self._roi_manager.load()   # safe — returns False if file missing
    return self._roi_manager
```

The same `ROIManager` instance is reused for "Load ROIs", "Start", etc.

---

## Window Close Handler

```python
def closeEvent(self, event) -> None:
    if self._pipeline_thread and self._pipeline_thread.isRunning():
        self._pipeline_thread.stop()
        self._pipeline_thread.wait(3000)   # Wait up to 3 seconds for clean exit
    event.accept()
```

---

## Imports

```python
import sys
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import yaml
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QAction, QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QSizePolicy,
    QStatusBar, QTextEdit, QToolBar, QVBoxLayout, QWidget,
)
from classifier import SpaceClassifier
from detector import VehicleDetector
from pipeline import PipelineThread
from roi_manager import ROIManager
from tracker import CentroidTracker
from video_handler import VideoHandler
```
