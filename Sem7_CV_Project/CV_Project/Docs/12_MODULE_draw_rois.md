# 12 — Module: `draw_rois.py`

**File:** `parking_counter/draw_rois.py` (220 lines)
**Type:** Standalone script — does NOT import any other project modules

---

## Purpose

An interactive OpenCV window tool that lets the user click parking-space polygon vertices
on a video frame and save the resulting coordinates to a JSON file.

**Critical design constraint:** Must resize the video frame to the **same** `process_width`
as the main pipeline — otherwise saved coordinates will be at a different scale and won't
map correctly onto pipeline frames.

---

## CLI Interface

```bash
python draw_rois.py --video path/to/video.mp4
python draw_rois.py --video path/to/video.mp4 --output assets/parking_spots.json
python draw_rois.py --video path/to/video.mp4 --frame 120   # seek to frame 120
python draw_rois.py --video path/to/video.mp4 --frame 0 --config config.yaml
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--video` | Yes | — | Path to the video file |
| `--output` | No | `"assets/parking_spots.json"` | Output JSON file path |
| `--frame` | No | `0` | Frame index to use as background (seek position) |
| `--config` | No | `"config.yaml"` | Path to config file (read for `process_width`) |

---

## Global State Variables

```python
_current_polygon: List[Tuple[int, int]] = []   # Vertices being drawn (in progress)
_spots: List[List[Tuple[int, int]]] = []        # All completed polygons
_frame_display: np.ndarray | None = None        # Background frame (frozen video snapshot)
```

These are module-level globals modified by the mouse callback and the main loop.
This is intentional — OpenCV mouse callbacks can't use `self`.

---

## Key Function: `_load_process_width(config_path) → int`

```python
def _load_process_width(config_path: str = "config.yaml") -> int:
    try:
        with open(config_path, "r") as fh:
            cfg = yaml.safe_load(fh)
        return int(cfg.get("video", {}).get("process_width", 640))
    except Exception:
        return 640  # Safe default if config is missing or malformed
```

This is the **resolution synchronisation mechanism**. By reading the same `process_width`
from `config.yaml`, `draw_rois.py` always draws on a canvas that is pixel-identical
to what `VideoHandler` produces in the pipeline.

---

## Main Function: `run(video_path, output_path, seek_frame=0, config_path="config.yaml")`

```python
def run(video_path, output_path, seek_frame=0, config_path="config.yaml"):
    # 1. Open video
    cap = cv2.VideoCapture(video_path)
    if seek_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, seek_frame)
    ret, frame = cap.read()
    cap.release()

    # 2. Resize to process_width (CRITICAL for coordinate accuracy)
    process_width = _load_process_width(config_path)
    h, w = frame.shape[:2]
    if w != process_width:
        scale = process_width / w
        frame = cv2.resize(frame, (process_width, int(h * scale)),
                           interpolation=cv2.INTER_LINEAR)

    _frame_display = frame.copy()

    # 3. Create OpenCV window and set mouse callback
    window = "Draw Parking ROIs"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, _mouse_callback)

    # 4. Interactive loop
    while True:
        display = _draw_state(_frame_display)
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("u"):   _current_polygon.pop()   # undo vertex
        elif key == ord("d"): _spots.pop()              # delete last spot
        elif key == ord("s"): _save(output_path); break # save and quit
        elif key in (ord("q"), 27): break               # quit without saving (27=Escape)

    cv2.destroyAllWindows()
```

---

## Mouse Callback: `_mouse_callback(event, x, y, _flags, _param)`

```python
def _mouse_callback(event, x, y, _flags, _param):
    global _current_polygon

    if event == cv2.EVENT_LBUTTONDOWN:
        _current_polygon.append((x, y))      # Add vertex

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(_current_polygon) >= 3:
            _spots.append(list(_current_polygon))
            _current_polygon = []             # Reset for next polygon
        else:
            print("[draw_rois] Need at least 3 vertices to close a polygon.")
```

**Minimum vertices enforcement:** Right-click does nothing if fewer than 3 vertices exist.

---

## Display Function: `_draw_state(frame) → np.ndarray`

Renders the current state on a copy of the background frame:

```python
def _draw_state(frame: np.ndarray) -> np.ndarray:
    out = frame.copy()

    # Draw all completed spots
    for i, poly in enumerate(_spots):
        pts = np.array(poly, dtype=np.int32)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], (0, 200, 0))           # Green semi-transparent fill
        cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)   # 25% alpha
        cv2.polylines(out, [pts], isClosed=True, color=(0, 220, 0), thickness=2)
        # Label with spot index
        cx = int(np.mean([p[0] for p in poly]))
        cy = int(np.mean([p[1] for p in poly]))
        cv2.putText(out, str(i), (cx-6, cy+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    # Draw in-progress polygon
    if _current_polygon:
        for pt in _current_polygon:
            cv2.circle(out, pt, 4, (0, 165, 255), -1)       # Orange dots at vertices
        for j in range(1, len(_current_polygon)):
            cv2.line(out, _current_polygon[j-1], _current_polygon[j],
                     (0, 165, 255), 1)                        # Orange connecting lines

    # HUD instructions (top-left)
    instructions = [
        "Left-click: add vertex",
        "Right-click: complete spot",
        "'u': undo vertex  'd': delete spot",
        "'s': save & quit  'q': quit",
        f"Spots: {len(_spots)}   Vertices: {len(_current_polygon)}",
    ]
    for k, line in enumerate(instructions):
        cv2.putText(out, line, (8, 18 + k*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)

    return out
```

---

## Save Function: `_save(output_path: str) → None`

```python
def _save(output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = [{"id": i, "polygon": list(poly)} for i, poly in enumerate(_spots)]
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
```

IDs are assigned as sequential integers starting from 0.

---

## Output Format

```json
[
  {
    "id": 0,
    "polygon": [
      [120, 80],
      [200, 80],
      [210, 140],
      [115, 140]
    ]
  },
  {
    "id": 1,
    "polygon": [
      [205, 80],
      [285, 78],
      [295, 138],
      [212, 140]
    ]
  }
]
```

---

## Key Bindings Summary

| Key | Action |
|-----|--------|
| Left-click | Add vertex to current polygon |
| Right-click | Complete current polygon (requires ≥3 vertices) |
| `u` | Remove last vertex from current polygon |
| `d` | Delete the most recently completed spot |
| `s` | Save all spots to JSON and quit |
| `q` or `Escape` (key 27) | Quit without saving |

---

## Imports

```python
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import yaml
```

No imports from other project modules.

---

## How `ui.py` Launches This Script

```python
subprocess.Popen(
    [
        sys.executable, "draw_rois.py",
        "--video", self._video_source,
        "--output", self._cfg["roi"]["coordinates_file"],
        "--config", "config.yaml",
    ],
    cwd=str(Path(__file__).parent)  # Must run from parking_counter/
)
```

Running in a subprocess (not as a thread) is intentional: OpenCV's `imshow` must run
on the main thread of its process; running it from a Qt thread causes crashes.

---

## Common Mistakes to Avoid

1. **Do not run `draw_rois.py` without `--config`** — it will default to 640px but
   if your config has a different `process_width`, coordinates will be wrong.

2. **Always save with `s` key, not just closing the window** — closing without `s` discards
   all drawn polygons.

3. **At least 3 vertices per polygon** — right-clicking with fewer does nothing.

4. **The background frame is a single snapshot** (the first frame or `--frame` index).
   Park cars in the lot before recording so you have reference positions to draw on.
