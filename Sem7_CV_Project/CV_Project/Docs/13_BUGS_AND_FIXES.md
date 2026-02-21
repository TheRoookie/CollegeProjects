# 13 — Bugs Encountered & Fixes Applied

This document records every bug encountered during development and testing of the
Car Parking Space Counter, along with the root cause and exact fix.

---

## Bug 1: ROI Coordinates Are Off by 2×

**Symptom:** After drawing ROIs with `draw_rois.py` and loading them in the pipeline,
the coloured polygon overlays appear in the wrong positions — roughly double the expected
distance from the top-left.

**Root Cause:**
`draw_rois.py` used a hardcoded canvas width of **1280px**, while the pipeline resized
frames to **640px** (from `config.yaml → video.process_width`). All saved coordinates
were at 1280-pixel scale, but the pipeline expected 640-pixel scale. Result: every
coordinate was exactly 2× too large.

**Fix:**
`draw_rois.py` now reads `process_width` from `config.yaml` via `_load_process_width()`:

```python
def _load_process_width(config_path: str = "config.yaml") -> int:
    try:
        with open(config_path, "r") as fh:
            cfg = yaml.safe_load(fh)
        return int(cfg.get("video", {}).get("process_width", 640))
    except Exception:
        return 640
```

The frame is then resized to exactly `process_width` pixels wide before the drawing
canvas is shown. Saved coordinates now map 1:1 to pipeline frames.

**Files changed:** `draw_rois.py`

---

## Bug 2: Video Plays Extremely Slowly at `detection_fps: 1`

**Symptom:** A 10-second video took ~300 seconds to play through. Setting `max_fps: 1`
caused the entire video (display + detection) to run at 1 frame per second.

**Root Cause:**
`max_fps` in `VideoHandler` was being used as the sole FPS control. Setting
`max_fps=1` throttled BOTH display and detection to 1 fps. For a 30-fps video,
`time.sleep(1.0)` was called between every frame, giving 1/30th real-time speed.

**Fix:**
Added a **separate** `detection_fps` parameter to `config.yaml` and `PipelineThread`.

- `max_fps` (in `VideoHandler`) controls **display/playback speed**
- `detection_fps` (in `PipelineThread`) controls **how often YOLO runs**

New logic in `PipelineThread.run()`:
```python
video_fps: float = self._video.fps or 30.0
if self._detection_fps > 0:
    detect_every = max(1, round(video_fps / self._detection_fps))
else:
    detect_every = 1

frame_idx = 0
for _raw, frame in self._video.frames():
    run_detection = (frame_idx % detect_every == 0)
    annotated = self._process_frame(frame, run_detection=run_detection)
    frame_idx += 1
```

With `max_fps=30` and `detection_fps=1`, the GUI plays smoothly at 30fps while YOLO
runs only once per second.

**Files changed:** `config.yaml` (added `detection_fps: 1`), `pipeline.py`

---

## Bug 3: `cv2.error: Unknown C++ exception` in `GaussianBlur`

**Symptom:** Running the pipeline crashed with:
```
cv2.error: OpenCV(4.x.x) Unknown C++ exception from OpenCV code
```
at the `cv2.GaussianBlur` or `cv2.adaptiveThreshold` call inside `PixelCountClassifier`.

**Root Cause:**
`ROIManager.extract_patch()` could return an array with zero pixels in one or both
dimensions when the polygon's bounding box extended outside the frame or was degenerate.
`cv2.GaussianBlur` crashes on zero-dimension arrays.
`cv2.adaptiveThreshold` crashes when `block_size` ≥ the patch dimension.

Two separate issues:
1. `extract_patch()` returned a valid-looking but zero-height or zero-width slice
2. `PixelCountClassifier` had no guards for None/degenerate patches

**Fix — `roi_manager.py`:**
```python
if x2 <= x1 or y2 <= y1:
    return None   # Explicitly return None for degenerate crops
return masked[y1:y2, x1:x2]
```

**Fix — `classifier.py`:**
```python
if patch is None or patch.ndim < 2:
    return SpaceStatus.EMPTY

ph, pw = patch.shape[:2]
if ph == 0 or pw == 0:
    return SpaceStatus.EMPTY

# Clamp blur kernel to patch dimensions
ksize = min(self.blur_kernel, ph, pw)
ksize = ksize if ksize % 2 == 1 else max(1, ksize - 1)
blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)

# Clamp adaptive block to patch dimensions
block = min(self.adaptive_block, ph, pw)
block = block if block % 2 == 1 else max(1, block - 1)
if block < 3:
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
else:
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block, self.adaptive_c)
```

**Files changed:** `roi_manager.py`, `classifier.py`

---

## Bug 4: Occupied Space Not Updating to "Empty" When Car Leaves

**Symptom:** When a vehicle drove away from a parking space, the space remained
marked "occupied" even though the detection was gone. The status never updated.

**Root Cause:**
1. The `pixel_threshold` was 0.18 (too low) — shadows and ground markings on textured
   or brick surfaces triggered "occupied" even with no car present.
2. There was no temporal smoothing — a single "empty" frame should not immediately
   override several "occupied" frames (but neither should "occupied" never update).

**Fix:**
Added debounce mechanism to `ParkingSpot`:

```python
def update_status(self, new_status: str, debounce: int = 3) -> None:
    if new_status == self._candidate:
        self._candidate_count += 1
    else:
        self._candidate = new_status
        self._candidate_count = 1

    if self._candidate_count >= debounce:
        self.status = self._candidate
```

Raised `pixel_threshold` to 0.35 to reduce false "occupied" triggers from shadows/markings.

**Files changed:** `roi_manager.py`, `config.yaml`

---

## Bug 5: All Spots Permanently Showing "Available" (Inaccurate Classification)

**Symptom:** Even with cars clearly parked in spots, the system showed all spaces
as "Available" (empty). The pixel-count classifier was never triggering "occupied".

**Root Cause:**
The `pixel_count` method is fundamentally unreliable on brick, cobblestone, or
textured pavement. Adaptive thresholding produces many white pixels even from
ground texture alone, making the threshold meaningless.

**Fix:**
Replaced the primary classification method with **YOLO bbox overlap**:

New `_detection_overlap()` function in `pipeline.py`:
```python
def _detection_overlap(spot, bbox, frame_h, frame_w) -> float:
    spot_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.fillPoly(spot_mask, [spot.polygon], 255)

    det_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.rectangle(det_mask, (max(0,x1), max(0,y1)), (min(frame_w-1,x2), min(frame_h-1,y2)), 255, -1)

    spot_area = int(np.count_nonzero(spot_mask))
    if spot_area == 0:
        return 0.0
    intersection = int(np.count_nonzero(cv2.bitwise_and(spot_mask, det_mask)))
    return intersection / spot_area
```

Config updated: `method: "overlap"`, `overlap_threshold: 0.25`.

This reuses the YOLO detections already being computed for tracking, adds no
extra inference cost, and is robust to texture and lighting.

**Files changed:** `pipeline.py`, `config.yaml`

---

## Bug 6: `WinError 1114` / `c10.dll` DLL Load Failed on Startup

**Symptom:** Application crashed immediately on import with:
```
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed.
```
pointing to `c10.dll` within the PyTorch installation.

**Root Cause — Part 1:** CUDA-enabled `torch` package was installed (downloaded from
the default PyPI index). This build requires a specific NVIDIA GPU driver version.
The machine either had no GPU or an incompatible driver version.

**Root Cause — Part 2 (code bug):** Even after fixing Part 1, the crash persisted
because of a bug in `ui.py._start_pipeline()`:

```python
# WRONG — original code:
method = cls_cfg.get("method", "overlap")
if method == "pixel_count":
    classifier = SpaceClassifier(method="pixel_count", ...)
else:
    # This branch was hit for method="overlap"!
    classifier = SpaceClassifier(method="cnn", ...)  # imports torch → crash
```

When `method="overlap"`, the old code fell into the `else` branch and created a
`CNNClassifier`, which imported `torch`, which crashed with `WinError 1114`.

**Fix — Part 1 (install CPU torch):**
```powershell
pip install torch==2.2.2+cpu torchvision==0.17.2+cpu --index-url https://download.pytorch.org/whl/cpu --upgrade
```

Updated `requirements.txt`:
```
torch==2.2.2+cpu
torchvision==0.17.2+cpu
--extra-index-url https://download.pytorch.org/whl/cpu
```

**Fix — Part 2 (ui.py classifier selection):**
```python
method = cls_cfg.get("method", "overlap")
if method == "cnn":
    # CNN backend — only when explicitly requested
    classifier = SpaceClassifier(method="cnn", checkpoint_path=cls_cfg["cnn_checkpoint"])
else:
    # "overlap" and "pixel_count" both use pixel_count as SpaceClassifier backend
    # (torch is NEVER imported for these paths)
    classifier = SpaceClassifier(
        method="pixel_count",
        pixel_threshold=cls_cfg.get("pixel_threshold", 0.25),
        ...
    )
```

**Files changed:** `requirements.txt`, `ui.py`

---

## Bug 7: YOLO Silent Failure Causing All Spots to Show Available

**Symptom:** After fixing Bug 6, YOLO was still silently failing (returning 0 detections
rather than throwing) because the `_detector_failed` flag was being set based on
exceptions, but the underlying issue (wrong torch build) caused YOLO to return empty
results without raising.

**Root Cause:** With the CUDA torch build partially working, YOLO would initialise
but produce 0 detections silently. The overlap method therefore found no detections
and marked all spots as empty.

**Fix:** Properly resolved by installing CPU-only torch (Part 1 of Bug 6 fix).
With correct torch, YOLO runs on CPU and produces correct detections.

The `_detector_failed` flag and fallback remain in place as defensive coding.

**Files changed:** None (covered by Bug 6 fix)

---

## Summary Table

| # | Symptom | Root Cause | Files Changed |
|---|---------|-----------|---------------|
| 1 | ROIs in wrong position | draw_rois used 1280px, pipeline used 640px | `draw_rois.py` |
| 2 | Video plays at 1/300th speed | `max_fps` throttled both display and detection | `config.yaml`, `pipeline.py` |
| 3 | cv2 C++ exception crash | Zero-dimension patch passed to GaussianBlur | `roi_manager.py`, `classifier.py` |
| 4 | Status never updates to empty | No debounce + threshold too low | `roi_manager.py`, `config.yaml` |
| 5 | All spots show available | pixel_count unreliable on textured surfaces | `pipeline.py`, `config.yaml` |
| 6 | WinError 1114 c10.dll | CUDA torch + ui.py CNNClassifier bug | `requirements.txt`, `ui.py` |
| 7 | YOLO silent empty results | CUDA torch partially functional | (resolved by Bug 6 fix) |
