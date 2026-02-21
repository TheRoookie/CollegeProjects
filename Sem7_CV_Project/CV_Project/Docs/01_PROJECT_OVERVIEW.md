# 01 — Project Overview

## What This Project Is

**Car Parking Space Counter** — a desktop computer-vision application that:

1. Takes a video file (or RTSP stream) of a car-park recorded from above.
2. Lets the user interactively draw polygon regions-of-interest (ROIs) over each parking space.
3. Runs YOLOv8 nano to detect vehicles in every Nth frame.
4. Tracks detected vehicles across frames using a centroid-matching algorithm.
5. Determines which parking spaces are **occupied** or **empty** by computing the geometric overlap between each YOLO detection bounding box and each space polygon.
6. Displays the live annotated feed in a PyQt5 dark-theme GUI with a real-time analytics sidebar.

---

## Academic Context

This is a **Semester 7 Computer Vision course project**.
Six distinct CV syllabus topics are each implemented in a dedicated module:

| # | Syllabus Topic | Module | Implementation |
|---|----------------|--------|----------------|
| 1 | **Video Processing** | `video_handler.py` | `cv2.VideoCapture`, FPS cap, resize, loop |
| 2 | **Image Segmentation** | `roi_manager.py` | `cv2.fillPoly` polygon masking per space |
| 3 | **Image Filtering** | `classifier.py` | Gaussian blur → adaptive thresholding |
| 4 | **Object Detection** | `detector.py` | YOLOv8 nano (COCO: car/motorcycle/bus/truck) |
| 5 | **Object Tracking** | `tracker.py` | Centroid tracker + Hungarian algorithm |
| 6 | **Image Classification** | `classifier.py` / `pipeline.py` | Pixel-count binary classifier + overlap method |

---

## Technology Stack

| Library | Version (exact tested) | Role |
|---------|------------------------|------|
| Python | 3.14.0 | Runtime |
| opencv-python | ≥ 4.8 (tested 4.13.0) | All image I/O + drawing |
| numpy | ≥ 1.24 (tested 2.4.2) | Array maths |
| ultralytics | ≥ 8.0 | YOLOv8 model loader & inference |
| PyQt5 | ≥ 5.15 | GUI framework |
| PyYAML | ≥ 6.0 | Configuration file parsing |
| scipy | ≥ 1.11 (tested 1.17.0) | `linear_sum_assignment` for tracker |
| Pillow | ≥ 10.0 | Image format conversion in CNN path |
| torch | 2.2.2+cpu | CNN classifier (optional) |
| torchvision | 0.17.2+cpu | CNN transforms (optional) |

> **CRITICAL:** Install **CPU-only** PyTorch from the whl index below.
> CUDA builds cause `WinError 1114 c10.dll` on machines without a matching GPU driver.

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        main.py (entry)                        │
│   argparse  →  QApplication  →  MainWindow(config.yaml)      │
└──────────────────────┬───────────────────────────────────────┘
                       │ QThread.start()
┌──────────────────────▼───────────────────────────────────────┐
│                   PipelineThread (pipeline.py)                │
│                                                              │
│  VideoHandler.frames()                                        │
│       │  every Nth frame (detect_every)                      │
│       ▼                                                       │
│  VehicleDetector.detect()  ──→  CentroidTracker.update()     │
│       │                                                       │
│       ▼  for each ParkingSpot:                               │
│  _detection_overlap(spot, bbox)  ≥ overlap_threshold?        │
│       │  yes → "occupied"   no → "empty"                     │
│       ▼                                                       │
│  ParkingSpot.update_status(new_status, debounce)             │
│       │                                                       │
│       ▼                                                       │
│  ROIManager.draw_overlays()  +  tracker trail drawing        │
│       │                                                       │
│  emit frame_ready(annotated_frame)                           │
│  emit stats_updated(total, empty, occupied)                  │
└──────────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
  VideoLabel (QLabel)         AnalyticsSidebar
  (scaled live feed)          (counts, bar, log)
```

---

## Data Flow Summary

```
Video file
  └─► VideoHandler yields (raw_frame, processed_frame) at max_fps
        └─► Every detect_every-th frame:
              VehicleDetector → List[Detection(bbox, confidence, class_id)]
              CentroidTracker → Dict[id, Track(centroid, history, ...)]
              For each ParkingSpot:
                _detection_overlap() → ratio in [0,1]
                ratio ≥ 0.25 → "occupied" else "empty"
                ParkingSpot.update_status(debounce=2)
        └─► Non-detection frames: reuse last statuses + cached detections
        └─► ROIManager.draw_overlays() on processed_frame
        └─► emit annotated frame + stats to GUI
```

---

## Directory Layout

```
CV_Project/
├── parking_counter/          ← All Python source + config
│   ├── config.yaml           ← Master configuration (all tunable params)
│   ├── requirements.txt      ← Exact pip dependencies
│   ├── main.py               ← Entry point (argparse + QApplication)
│   ├── ui.py                 ← PyQt5 main window + widgets
│   ├── pipeline.py           ← QThread orchestrating all CV modules
│   ├── video_handler.py      ← VideoHandler: VideoCapture wrapper
│   ├── roi_manager.py        ← ROIManager + ParkingSpot dataclass
│   ├── detector.py           ← VehicleDetector (YOLOv8)
│   ├── tracker.py            ← CentroidTracker (Hungarian matching)
│   ├── classifier.py         ← PixelCountClassifier + CNNClassifier
│   ├── draw_rois.py          ← OpenCV interactive ROI drawing tool
│   └── assets/
│       ├── yolov8n.pt        ← YOLO weights (auto-downloaded if missing)
│       └── parking_spots.json← ROI polygons (generated by draw_rois.py)
├── Docs/                     ← Specification documents (this folder)
├── README.md                 ← User-facing documentation
├── implementation.md         ← Original professor spec
└── Requriments.md            ← Course requirements
```

---

## Running the App

```bash
# From inside parking_counter/ with .venv activated:
cd parking_counter
.venv\Scripts\activate         # Windows
python main.py
```

Workflow inside the GUI:
1. **Open Video** — select `.mp4`/`.avi`/`.mkv`/`.mov`/`.webm`
2. **Draw ROIs** — launches `draw_rois.py`, draw polygons, press `s` to save
3. **Load ROIs** — load `assets/parking_spots.json`
4. **Start** — begins the pipeline; sidebar shows live counts
5. **Stop** — halts the pipeline

---

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| `method: "overlap"` as default classifier | Pixel-count is unreliable on textured/brick surfaces; YOLO overlap uses already-computed detection results |
| `detection_fps: 1` separate from `max_fps: 30` | Keeps GUI at 30 fps while running expensive YOLO only once per second |
| `status_debounce: 2` | Prevents single-frame flicker when a car enters/leaves |
| CPU-only PyTorch | CUDA build crashes on systems without matching GPU driver (WinError 1114) |
| `PipelineThread(QThread)` | Keeps GUI responsive; signals/slots cross the thread boundary safely |
| `_detector_failed` flag | Auto-falls back to pixel_count if YOLO throws; logs error only once |
