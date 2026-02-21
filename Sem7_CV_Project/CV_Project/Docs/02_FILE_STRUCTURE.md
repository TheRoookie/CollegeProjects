# 02 — File Structure

## Complete Directory Tree

```
CV_Project/                          ← Workspace root
│
├── parking_counter/                 ← All application code lives here
│   │
│   ├── .venv/                       ← Python virtual environment (Windows)
│   │   └── Scripts/activate         ← Activation script
│   │
│   ├── assets/                      ← Runtime assets
│   │   ├── yolov8n.pt               ← YOLOv8 nano weights (auto-downloaded by ultralytics)
│   │   └── parking_spots.json       ← ROI coordinates (created by draw_rois.py)
│   │
│   ├── config.yaml                  ← MASTER CONFIG — all tunable parameters
│   ├── requirements.txt             ← Exact pip dependencies
│   │
│   ├── main.py                      ← CLI entry point, argparse, QApplication launch
│   ├── ui.py                        ← PyQt5 GUI: MainWindow, VideoLabel, AnalyticsSidebar
│   ├── pipeline.py                  ← QThread: orchestrates all CV modules per frame
│   ├── video_handler.py             ← VideoHandler: cv2.VideoCapture wrapper
│   ├── roi_manager.py               ← ROIManager + ParkingSpot dataclass
│   ├── detector.py                  ← VehicleDetector: YOLOv8 wrapper + Detection dataclass
│   ├── tracker.py                   ← CentroidTracker: Hungarian-algorithm tracker + Track dataclass
│   ├── classifier.py                ← PixelCountClassifier + CNNClassifier + SpaceClassifier façade
│   └── draw_rois.py                 ← Standalone OpenCV tool for labelling ROI polygons
│
├── Docs/                            ← Specification documents (this folder)
│   ├── 01_PROJECT_OVERVIEW.md
│   ├── 02_FILE_STRUCTURE.md         ← You are here
│   ├── 03_ENVIRONMENT_SETUP.md
│   ├── 04_CONFIG_REFERENCE.md
│   ├── 05_MODULE_video_handler.md
│   ├── 06_MODULE_roi_manager.md
│   ├── 07_MODULE_detector.md
│   ├── 08_MODULE_tracker.md
│   ├── 09_MODULE_classifier.md
│   ├── 10_MODULE_pipeline.md
│   ├── 11_MODULE_ui.md
│   ├── 12_MODULE_draw_rois.md
│   ├── 13_BUGS_AND_FIXES.md
│   └── 14_DATA_FORMATS.md
│
├── README.md                        ← User-facing documentation
├── implementation.md                ← Professor-supplied feature spec
└── Requriments.md                   ← Course requirements document
```

---

## File Roles (one-line summary)

| File | Purpose |
|------|---------|
| `config.yaml` | Single source of truth for ALL tunable parameters |
| `requirements.txt` | Exact pinned pip packages; includes CPU-torch install hint |
| `main.py` | Parses `--config` arg; creates `QApplication`; shows `MainWindow` |
| `ui.py` | Full PyQt5 GUI: toolbar, video display, analytics sidebar, pipeline wiring |
| `pipeline.py` | `QThread` that ties VideoHandler → Detector → Tracker → Classifier → draw; emits signals |
| `video_handler.py` | Wraps `cv2.VideoCapture`; resizes frames; enforces FPS cap; loops files |
| `roi_manager.py` | Loads/saves JSON ROIs; `ParkingSpot` dataclass with debounce; polygon masking |
| `detector.py` | Lazy-loads YOLOv8 nano; returns `List[Detection]` filtered to vehicle COCO classes |
| `tracker.py` | Centroid tracker; builds cost matrix with `cdist`; solves with `linear_sum_assignment` |
| `classifier.py` | `PixelCountClassifier` (adaptive threshold), `CNNClassifier` (PyTorch), `SpaceClassifier` façade |
| `draw_rois.py` | Standalone OpenCV window; mouse-click polygon builder; saves `parking_spots.json` |
| `assets/yolov8n.pt` | YOLOv8 nano COCO weights — auto-downloaded by ultralytics if absent |
| `assets/parking_spots.json` | Polygon coordinates for each parking space (created by `draw_rois.py`) |

---

## Import Dependency Graph

```
main.py
  └─ ui.py
       ├─ classifier.py  (SpaceClassifier)
       ├─ detector.py    (VehicleDetector)
       ├─ pipeline.py    (PipelineThread)
       │    ├─ video_handler.py   (VideoHandler)
       │    ├─ roi_manager.py     (ROIManager, ParkingSpot)
       │    ├─ detector.py        (VehicleDetector, Detection)
       │    ├─ tracker.py         (CentroidTracker)
       │    └─ classifier.py      (SpaceClassifier, SpaceStatus)
       ├─ roi_manager.py  (ROIManager)
       ├─ tracker.py      (CentroidTracker)
       └─ video_handler.py (VideoHandler)

draw_rois.py  (standalone — no imports from other project files)
```

No circular imports exist. `draw_rois.py` is entirely standalone.

---

## Configuration Loading Chain

```
config.yaml
  ├─ Read by main.py → passed as config_path string to MainWindow
  ├─ Read by MainWindow.__init__ via yaml.safe_load
  │    └─ cfg dict keys: video / roi / detection / tracker / classifier / display
  └─ Read independently by draw_rois._load_process_width() for canvas size sync
```

---

## Key Assets

### `assets/yolov8n.pt`
- YOLOv8 nano COCO-pretrained weights
- Size: ~6 MB
- Auto-downloaded by ultralytics on first run if file is absent
- Path configurable via `config.yaml → detection.model_weights`

### `assets/parking_spots.json`
- JSON array of polygon objects
- Created by `draw_rois.py`
- Loaded by `ROIManager.load()`
- Must use **same resolution** as `config.yaml → video.process_width` (640 px default)
- Full schema documented in [14_DATA_FORMATS.md](14_DATA_FORMATS.md)
