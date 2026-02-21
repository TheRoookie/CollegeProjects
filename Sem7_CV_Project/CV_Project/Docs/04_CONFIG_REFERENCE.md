# 04 — Config Reference (`config.yaml`)

## Complete File (Current Working Version)

```yaml
# config.yaml
# -----------
# Master configuration file for the Car Parking Space Counter.
# All modules read their settings from this file via MainWindow._cfg.
# draw_rois.py reads only video.process_width independently.

video:
  process_width: 640      # Resize every frame to this width (height scales proportionally)
  max_fps: 30             # Maximum display FPS — controls GUI playback smoothness
  detection_fps: 1        # How many times per second to run YOLO detection
                          # detect_every = round(video_fps / detection_fps)
                          # e.g. 30fps video, detection_fps=1 → detect every 30th frame

roi:
  coordinates_file: "assets/parking_spots.json"   # Path to saved ROI polygons

detection:
  model_weights: "assets/yolov8n.pt"   # Path to YOLO weights (auto-downloaded if absent)
  confidence_threshold: 0.45           # Minimum YOLO confidence to keep a detection
  iou_threshold: 0.45                  # NMS IoU threshold for YOLO
  vehicle_classes: [2, 3, 5, 7]        # COCO class IDs: car=2 motorcycle=3 bus=5 truck=7
  device: "cpu"                        # Inference device: "cpu" | "cuda" | "mps"

tracker:
  max_distance: 50          # Max pixel distance for centroid matching across frames
  max_disappeared: 20       # Frames a track can be absent before being deregistered

classifier:
  method: "overlap"         # "overlap" (recommended) | "pixel_count" | "cnn"
  overlap_threshold: 0.25   # Fraction of spot polygon that must be covered by a detection
                            # to call that spot "occupied". Range: 0.0–1.0
  pixel_threshold: 0.25     # For pixel_count fallback: non-zero pixel ratio threshold
  blur_kernel: 5            # Gaussian blur kernel size (must be odd)
  adaptive_block: 11        # adaptiveThreshold block size (must be odd, ≥3)
  adaptive_c: 2             # adaptiveThreshold constant C
  status_debounce: 2        # Consecutive frames that must agree before status changes

display:
  empty_color: [0, 200, 0]      # BGR color for empty spots (green)
  occupied_color: [0, 0, 220]   # BGR color for occupied spots (red)
  unknown_color: [200, 200, 0]  # BGR color for unknown spots (yellow)
  box_thickness: 2              # Polygon outline thickness in pixels
  font_scale: 0.5               # Font scale for spot ID labels
```

---

## Parameter Reference Table

### `video` section

| Key | Type | Default | Range / Valid Values | Effect |
|-----|------|---------|---------------------|--------|
| `process_width` | int | 640 | 320–1920 (even integers) | Width to resize all frames to. Height auto-scaled. Must match the width used in `draw_rois.py` — both read this same value. |
| `max_fps` | int | 30 | 1–120, 0=unlimited | Maximum frames per second the GUI renders. Does NOT affect detection frequency. |
| `detection_fps` | float | 1 | 0.1–video_fps, 0=every frame | Frequency of YOLO detection runs. `detect_every = round(video_fps / detection_fps)`. Lower values = faster processing but less frequent updates. |

### `roi` section

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `coordinates_file` | str | `"assets/parking_spots.json"` | Relative to `parking_counter/` working directory. Created by `draw_rois.py`. |

### `detection` section

| Key | Type | Default | Range / Valid Values | Effect |
|-----|------|---------|---------------------|--------|
| `model_weights` | str | `"assets/yolov8n.pt"` | Any valid ultralytics YOLO path | Path to YOLOv8 weights. Ultralytics auto-downloads if file missing. |
| `confidence_threshold` | float | 0.45 | 0.01–0.99 | Minimum YOLO confidence score to include a detection. Higher = fewer false positives. |
| `iou_threshold` | float | 0.45 | 0.01–0.99 | NMS IoU threshold. Lower keeps more overlapping boxes. |
| `vehicle_classes` | list[int] | [2, 3, 5, 7] | COCO class IDs | Which object classes count as vehicles. 2=car, 3=motorcycle, 5=bus, 7=truck. |
| `device` | str | `"cpu"` | `"cpu"`, `"cuda"`, `"mps"` | Inference device. Always use `"cpu"` on Windows without compatible NVIDIA GPU. |

### `tracker` section

| Key | Type | Default | Range | Effect |
|-----|------|---------|-------|--------|
| `max_distance` | int | 50 | 10–200 | Maximum centroid-to-centroid distance (pixels) to consider two detections the same vehicle. Larger = tracks survive bigger jumps. |
| `max_disappeared` | int | 20 | 1–100 | Number of consecutive frames a track may be absent (e.g., due to occlusion) before being permanently removed. |

### `classifier` section

| Key | Type | Default | Range / Valid Values | Effect |
|-----|------|---------|---------------------|--------|
| `method` | str | `"overlap"` | `"overlap"`, `"pixel_count"`, `"cnn"` | Primary classification method. `"overlap"` uses YOLO bbox geometry. `"pixel_count"` uses adaptive threshold. `"cnn"` requires a checkpoint. |
| `overlap_threshold` | float | 0.25 | 0.05–0.95 | *overlap method only.* Fraction of the spot polygon area that a YOLO detection must cover for the spot to be "occupied". Lower = more sensitive (more false occupied), higher = more strict. |
| `pixel_threshold` | float | 0.25 | 0.05–0.90 | *pixel_count fallback.* Fraction of white pixels after adaptive threshold above which spot is "occupied". |
| `blur_kernel` | int | 5 | 3, 5, 7, 9 (odd) | Gaussian blur kernel size before adaptive threshold. Larger = more smoothing. |
| `adaptive_block` | int | 11 | 3, 5, 7, 9, 11 (odd, ≥3) | Neighbourhood size for adaptive threshold. Classifier code auto-clamps to patch size. |
| `adaptive_c` | int | 2 | 0–10 | Constant subtracted from adaptive threshold mean. Higher = less sensitive. |
| `status_debounce` | int | 2 | 1–10 | Number of consecutive frames that must agree on a status before it is committed. 1 = instant update (may flicker), 3+ = stable but sluggish. |

### `display` section

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `empty_color` | [B, G, R] | [0, 200, 0] | BGR, not RGB! Green for available spots. |
| `occupied_color` | [B, G, R] | [0, 0, 220] | Red for occupied spots. |
| `unknown_color` | [B, G, R] | [200, 200, 0] | Yellow for spots in unknown state (before first detection). |
| `box_thickness` | int | 2 | Line thickness for polygon outlines. |
| `font_scale` | float | 0.5 | Size of the spot ID number label rendered on each polygon. |

---

## How `detection_fps` Works

```
video_fps  = cap.get(cv2.CAP_PROP_FPS)  # e.g. 30.0
detect_every = max(1, round(video_fps / detection_fps))

frame_idx 0:  run_detection = (0 % 30 == 0) = True   ← YOLO runs
frame_idx 1:  run_detection = (1 % 30 == 0) = False  ← skip, reuse last
...
frame_idx 30: run_detection = (30 % 30 == 0) = True  ← YOLO runs again
```

Setting `detection_fps: 0` makes `detect_every = 1` (YOLO on every frame).

---

## Classification Method Decision Tree

```
classifier.method in config.yaml:

"overlap"     → use_detection_overlap=True  in PipelineThread
                YOLO runs → for each spot, compute bbox∩polygon fraction
                if fraction ≥ overlap_threshold → "occupied"
                if YOLO crashes → _detector_failed=True → fall back to pixel_count

"pixel_count" → use_detection_overlap=False in PipelineThread
                for each spot: extract_patch → GaussianBlur → adaptiveThreshold → count pixels
                if ratio ≥ pixel_threshold → "occupied"

"cnn"         → use_detection_overlap=False
                for each spot: extract_patch → resize(64×64) → CNN forward pass → argmax
                REQUIRES: a trained checkpoint at cls_cfg["cnn_checkpoint"]
```

---

## Critical Implementation Notes

1. **`empty_color`, `occupied_color`, `unknown_color` are BGR not RGB.**
   `[0, 200, 0]` = green (B=0, G=200, R=0). This is because OpenCV uses BGR.

2. **`process_width` must be identical between config.yaml and draw_rois.py.**
   `draw_rois.py` reads this value via `_load_process_width()`. If they differ, ROI coordinates
   will be off by a scale factor. Never manually set different values.

3. **`device: "cpu"` is mandatory on Windows without a compatible NVIDIA GPU.**
   Even if you have a GPU, if the CUDA driver version does not exactly match the torch
   build, you will get `WinError 1114 c10.dll`. Stay on `"cpu"` unless you have verified
   CUDA compatibility.

4. **All paths in config.yaml are relative to `parking_counter/` (the working directory).**
   `python main.py` must be run from inside `parking_counter/`, not from `CV_Project/`.
