# 05 — Module: `video_handler.py`

**Syllabus Topic:** Video Processing
**File:** `parking_counter/video_handler.py` (146 lines)

---

## Purpose

Wraps `cv2.VideoCapture` to provide:
1. A clean context-manager interface (`with VideoHandler(...) as vh:`)
2. A generator that yields `(original_frame, processed_frame)` pairs
3. Automatic aspect-ratio-preserving resize to a consistent `process_width`
4. A maximum-FPS cap via `time.sleep` (controls GUI playback rate — separate from `detection_fps`)
5. Automatic looping when a file-based source ends

---

## Class: `VideoHandler`

### Constructor

```python
VideoHandler(
    source: str | int,
    process_width: int = 640,
    max_fps: int = 30,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | str \| int | — | File path, RTSP URL, or integer webcam index (0=default camera) |
| `process_width` | int | 640 | Target width for resized frames. Must match `config.yaml → video.process_width` |
| `max_fps` | int | 30 | Max frames per second to yield. Implemented via `time.sleep`. `0` = no cap |

**Internal state after `__init__`:**
- `self._cap = None` (uninitialised)
- `self._frame_delay = 1.0 / max_fps` if `max_fps > 0`, else `0.0`

---

### Public Methods

#### `open() → None`
Opens the capture device. Raises `IOError` if the source cannot be opened.

```python
self._cap = cv2.VideoCapture(self.source)
if not self._cap.isOpened():
    raise IOError(f"Cannot open video source: {self.source!r}")
```

#### `release() → None`
Releases the capture resource. Sets `self._cap = None`.

#### `frames() → Generator[(np.ndarray, np.ndarray)]`
The main data generator. Must call `open()` first or it raises `RuntimeError`.

**Yields:** `(original_frame, processed_frame)` where:
- `original_frame` — raw BGR frame straight from the capture
- `processed_frame` — resized to `process_width` (height scaled proportionally)

**Loop behaviour:**
```
while True:
    ret, frame = cap.read()
    if not ret:
        if source is a file:
            seek back to frame 0 and retry once
            if still not ret: break
        else:
            break   (live stream ended)
    processed = _resize(frame)
    # FPS cap: sleep if needed to maintain frame_delay
    yield (frame, processed)
```

**FPS cap implementation:**
```python
elapsed = time.time() - last_time
sleep_for = self._frame_delay - elapsed
if sleep_for > 0:
    time.sleep(sleep_for)
last_time = time.time()
```

#### `_resize(frame: np.ndarray) → np.ndarray` (private)
```python
h, w = frame.shape[:2]
if w == self.process_width:
    return frame
scale = self.process_width / w
new_h = int(h * scale)
return cv2.resize(frame, (self.process_width, new_h), interpolation=cv2.INTER_LINEAR)
```
- Only resizes if the width differs from `process_width`
- Uses `cv2.INTER_LINEAR` (bilinear interpolation — good balance of quality/speed)

---

### Properties

| Property | Type | Returns |
|----------|------|---------|
| `fps` | float | `cap.get(cv2.CAP_PROP_FPS)` — native video FPS (0 if not opened) |
| `total_frames` | int | `cap.get(cv2.CAP_PROP_FRAME_COUNT)` — total frames (0 for live streams) |
| `original_size` | tuple(int, int) | `(width, height)` of the raw source before resize |

---

### Context Manager

```python
def __enter__(self) -> "VideoHandler":
    self.open()
    return self

def __exit__(self, *_: object) -> None:
    self.release()
```

Usage:
```python
with VideoHandler("video.mp4", process_width=640, max_fps=30) as vh:
    for original, processed in vh.frames():
        # process `processed`...
```

---

## Imports

```python
import time
from pathlib import Path
from typing import Generator, Optional, Tuple
import cv2
import numpy as np
```

---

## How `pipeline.py` Uses This Module

```python
# In PipelineThread.__init__:
self._video = video_handler  # passed in

# In PipelineThread.run():
self._video.open()
video_fps: float = self._video.fps or 30.0
detect_every = max(1, round(video_fps / self._detection_fps))

for _raw, frame in self._video.frames():
    if not self._running:
        break
    annotated = self._process_frame(frame, run_detection=...)
    self.frame_ready.emit(annotated)

self._video.release()
```

Note: `_raw` (the unreasized original) is ignored in the current pipeline. All processing uses `frame` (the resized version).

---

## Edge Cases and Guarantees

| Scenario | Behaviour |
|----------|-----------|
| File ends | Auto-seeks to frame 0 and loops indefinitely |
| Live stream loses connection | Generator stops, `finished_signal` emitted |
| `frames()` called before `open()` | Raises `RuntimeError("Call open() before iterating frames.")` |
| `max_fps=0` | `_frame_delay=0.0`, no sleep, runs as fast as possible |
| Frame width already equals `process_width` | `_resize` returns original array unchanged (no copy) |
| Frame data from `VideoCapture` is always BGR | True for file sources; true for most webcams |
