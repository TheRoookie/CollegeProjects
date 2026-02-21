# 14 — Data Formats

This document specifies every external data format used by the application:
1. `parking_spots.json` — ROI polygon coordinates
2. `config.yaml` — full annotated example
3. Internal data structures (for reference)

---

## Format 1: `parking_spots.json`

### Location
`parking_counter/assets/parking_spots.json`
(Configurable via `config.yaml → roi.coordinates_file`)

### Schema

**Type:** JSON Array of objects
**Created by:** `draw_rois.py`
**Read by:** `ROIManager.load()`

```json
[
  {
    "id": <integer>,
    "polygon": [
      [<x: integer>, <y: integer>],
      [<x: integer>, <y: integer>],
      ...
    ]
  },
  ...
]
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `id` | non-negative integer | Sequential ID starting from 0. Must be unique within the file. |
| `polygon` | array of [x, y] pairs | Vertex coordinates in pixel space. Minimum 3 vertices. |
| `polygon[i][0]` | integer | X coordinate (pixels from left edge) |
| `polygon[i][1]` | integer | Y coordinate (pixels from top edge) |

### Coordinate Space

- Origin: top-left corner of the frame
- X increases left → right
- Y increases top → bottom
- Units: pixels at `process_width` resolution (default: 640px wide)
- Height: proportional — a 1920×1080 video resized to 640px wide → 360px tall

**CRITICAL:** Coordinates must be created at the same resolution as `process_width` in
`config.yaml`. If you change `process_width`, you must re-draw all ROIs.

### Validity Rules

- Minimum 3 vertices per polygon (fewer cannot form a closed polygon)
- Coordinates should be within frame bounds (x: 0–`process_width`, y: 0–frame_height)
- `id` values must be unique; ideally sequential from 0 (auto-assigned by `draw_rois.py`)
- Polygon should be convex or non-self-intersecting for best masking results

### Example: 4-spot parking lot

```json
[
  {
    "id": 0,
    "polygon": [
      [23, 85],
      [97, 83],
      [101, 142],
      [22, 145]
    ]
  },
  {
    "id": 1,
    "polygon": [
      [100, 82],
      [174, 80],
      [178, 140],
      [103, 141]
    ]
  },
  {
    "id": 2,
    "polygon": [
      [176, 79],
      [251, 77],
      [255, 138],
      [179, 139]
    ]
  },
  {
    "id": 3,
    "polygon": [
      [253, 76],
      [327, 74],
      [331, 136],
      [256, 138]
    ]
  }
]
```

### How It Is Loaded

```python
# In ROIManager.load():
with open(self.coordinates_file, "r", encoding="utf-8") as fh:
    data: List[Dict] = json.load(fh)

self.spots = [
    ParkingSpot(
        spot_id=entry["id"],
        polygon=np.array(entry["polygon"], dtype=np.int32),
    )
    for entry in data
]
```

The polygon is converted to `np.ndarray` with `dtype=np.int32` for OpenCV compatibility.

---

## Format 2: `config.yaml` (Full Annotated)

```yaml
# config.yaml — complete annotated reference

video:
  # process_width: integer (pixels)
  # ALL frames are resized to this width. Height scales proportionally.
  # MUST match the canvas used in draw_rois.py (both read this value).
  process_width: 640

  # max_fps: integer (frames per second, 0 = unlimited)
  # Controls GUI display smoothness. Independent of detection frequency.
  max_fps: 30

  # detection_fps: float (detections per second)
  # YOLO runs every round(video_fps / detection_fps) frames.
  # Set to 0 to run YOLO on every frame.
  detection_fps: 1

roi:
  # coordinates_file: string (relative path from parking_counter/)
  # JSON file created by draw_rois.py.
  coordinates_file: "assets/parking_spots.json"

detection:
  # model_weights: string (path to .pt file)
  # Auto-downloaded by ultralytics if missing.
  model_weights: "assets/yolov8n.pt"

  # confidence_threshold: float [0.01, 0.99]
  # Minimum YOLO confidence score to keep a detection.
  confidence_threshold: 0.45

  # iou_threshold: float [0.01, 0.99]
  # Non-maxima suppression IoU threshold.
  iou_threshold: 0.45

  # vehicle_classes: list of COCO integer class IDs
  # 2=car, 3=motorcycle, 5=bus, 7=truck
  vehicle_classes: [2, 3, 5, 7]

  # device: string
  # "cpu" on Windows without verified CUDA GPU.
  # "cuda" or "mps" on GPU-enabled systems.
  device: "cpu"

tracker:
  # max_distance: integer (pixels)
  # Maximum centroid distance for cross-frame track matching.
  max_distance: 50

  # max_disappeared: integer (frames)
  # Frames a track can be absent before being permanently removed.
  max_disappeared: 20

classifier:
  # method: string
  # "overlap"     = YOLO bbox vs spot polygon intersection (RECOMMENDED)
  # "pixel_count" = adaptive threshold pixel ratio
  # "cnn"         = PyTorch CNN (requires trained checkpoint)
  method: "overlap"

  # overlap_threshold: float [0.05, 0.95]
  # Fraction of spot polygon covered by a detection → "occupied".
  # Only used when method="overlap".
  overlap_threshold: 0.25

  # pixel_threshold: float [0.05, 0.90]
  # Fraction of white pixels after adaptive threshold → "occupied".
  # Used as fallback when YOLO fails, or when method="pixel_count".
  pixel_threshold: 0.25

  # blur_kernel: odd integer [3, 9]
  # Gaussian blur kernel size. Applied before adaptive thresholding.
  blur_kernel: 5

  # adaptive_block: odd integer [3, 15]
  # Neighbourhood block size for adaptive thresholding.
  # Automatically clamped to patch dimensions by classifier code.
  adaptive_block: 11

  # adaptive_c: integer [0, 10]
  # Constant subtracted from local mean in adaptive threshold.
  adaptive_c: 2

  # status_debounce: integer [1, 10]
  # Consecutive frames that must agree before status change is committed.
  # 1 = instant (may flicker), 3+ = stable.
  status_debounce: 2

display:
  # Colors in BGR (not RGB!) format: [Blue, Green, Red], each 0-255.
  empty_color: [0, 200, 0]       # Green
  occupied_color: [0, 0, 220]    # Red
  unknown_color: [200, 200, 0]   # Yellow

  # box_thickness: integer (pixels)
  # Thickness of polygon outline strokes.
  box_thickness: 2

  # font_scale: float
  # Scale factor for spot ID number labels.
  font_scale: 0.5
```

---

## Format 3: Internal Data Structures

These are Python objects, not files, but are documented here for completeness.

### `Detection` (from `detector.py`)

```python
Detection(
    bbox=(x1: int, y1: int, x2: int, y2: int),  # top-left & bottom-right pixel coords
    confidence=0.87,                              # float [0, 1]
    class_id=2,                                   # COCO int (2=car, 3=moto, 5=bus, 7=truck)
    label="car",                                  # str
)
```

### `Track` (from `tracker.py`)

```python
Track(
    track_id=5,                                   # unique int, never reused
    centroid=(320, 240),                          # (cx, cy) pixel tuple
    bbox=(300, 220, 340, 260),                    # (x1, y1, x2, y2)
    disappeared=0,                                # int: frames absent
    history=[(318,238), (319,239), (320,240)],   # list of past centroids, max 50
)
```

### `ParkingSpot` (from `roi_manager.py`)

```python
ParkingSpot(
    spot_id=0,                                    # int
    polygon=np.array([[23,85],[97,83],...], dtype=np.int32),  # (N, 2) array
    status="empty",                               # "empty" | "occupied" | "unknown"
    # Private debounce state (not serialized):
    _bbox=(23, 83, 74, 62),                      # (x, y, w, h) — lazily cached
    _candidate="empty",                           # pending status
    _candidate_count=2,                           # consecutive agreement count
)
```

### Pipeline Signals

```python
PipelineThread.frame_ready     → (annotated_frame: np.ndarray)
PipelineThread.stats_updated   → (total: int, empty: int, occupied: int)
PipelineThread.error_occurred  → (message: str)
PipelineThread.finished_signal → ()
```

---

## Frame Coordinate System

```
(0, 0) ─────────────────────── (process_width-1, 0)
  │                                              │
  │         Video Frame                          │
  │         (BGR numpy array)                    │
  │                                              │
  │                                              │
(0, frame_height-1) ─────── (process_width-1, frame_height-1)
```

- All pixel coordinates are (x, y) where x=column, y=row
- OpenCV uses `frame[row, col]` = `frame[y, x]`
- All polygon coordinates, bounding boxes, and centroids use this system
- Default: `process_width=640`, `frame_height=360` for 16:9 source video
