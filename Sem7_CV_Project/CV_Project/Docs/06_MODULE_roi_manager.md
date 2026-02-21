# 06 — Module: `roi_manager.py`

**Syllabus Topic:** Image Segmentation
**File:** `parking_counter/roi_manager.py` (236 lines)

---

## Purpose

Manages the collection of parking-space polygon regions-of-interest (ROIs).
Responsibilities:
1. Store `ParkingSpot` objects (each with a polygon, an ID, and a status)
2. Load/save polygons to/from a JSON file
3. Extract a per-spot masked image patch from a video frame (segmentation)
4. Draw colour-coded polygon overlays on a display frame
5. Provide aggregate statistics (total, empty, occupied counts)

---

## Dataclass: `ParkingSpot`

```python
@dataclass
class ParkingSpot:
    spot_id: int
    polygon: np.ndarray          # shape (N, 2), dtype int32
    status: str = "unknown"      # "empty" | "occupied" | "unknown"
    _bbox: Optional[Tuple[int, int, int, int]] = field(default=None, repr=False)
    _candidate: str = field(default="unknown", repr=False)
    _candidate_count: int = field(default=0, repr=False)
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `spot_id` | int | Unique integer identifier (0-indexed, assigned at creation) |
| `polygon` | np.ndarray (N,2) int32 | Array of (x, y) vertex coordinates in pipeline resolution |
| `status` | str | Current committed status: `"empty"`, `"occupied"`, or `"unknown"` |
| `_bbox` | tuple or None | Cached bounding box — computed once by `bbox` property and reused |
| `_candidate` | str | The pending new status (debounce buffer) |
| `_candidate_count` | int | How many consecutive frames have agreed on `_candidate` |

### Method: `update_status(new_status, debounce=3)`

The debounce logic prevents single-frame flickering when a car enters or leaves.

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

**Behaviour examples (debounce=2):**

| Frame | `new_status` | `_candidate` | `_candidate_count` | `status` (committed) |
|-------|-------------|-------------|-------------------|---------------------|
| 1 | "occupied" | "occupied" | 1 | "unknown" (not yet) |
| 2 | "occupied" | "occupied" | 2 | "occupied" ✓ |
| 3 | "empty" | "empty" | 1 | "occupied" (not yet) |
| 4 | "occupied" | "occupied" | 1 | "occupied" (reset candidate) |
| 5 | "occupied" | "occupied" | 2 | "occupied" ✓ |

**Why debounce is critical:** On textured surfaces, a single frame may misclassify
a space. Requiring N consecutive agreement frames prevents rapid status flicker.

### Property: `bbox → Tuple[int, int, int, int]`

Returns `(x, y, w, h)` from `cv2.boundingRect(self.polygon)`.
Result is cached in `_bbox` so `cv2.boundingRect` is only computed once per spot.

```python
@property
def bbox(self) -> Tuple[int, int, int, int]:
    if self._bbox is None:
        x, y, w, h = cv2.boundingRect(self.polygon)
        self._bbox = (x, y, w, h)
    return self._bbox
```

### Property: `center → Tuple[int, int]`

Returns `(x + w//2, y + h//2)` — the centroid of the bounding box.

---

## Class: `ROIManager`

### Constructor

```python
ROIManager(coordinates_file: str | Path)
```

- `self.coordinates_file = Path(coordinates_file)`
- `self.spots: List[ParkingSpot] = []`

---

### Persistence Methods

#### `load() → bool`

Reads `coordinates_file` and populates `self.spots`.

```python
# Expected JSON format:
[
  {"id": 0, "polygon": [[x1,y1], [x2,y2], [x3,y3], ...]},
  {"id": 1, "polygon": [...]}
]
```

Creates one `ParkingSpot` per entry with `np.array(entry["polygon"], dtype=np.int32)`.
Returns `True` on success, `False` if the file does not exist.
Does **not** raise on missing file — caller gets `False` and decides what to do.

#### `save() → None`

Writes `self.spots` back to `coordinates_file`.
Creates parent directories if needed (`mkdir(parents=True, exist_ok=True)`).
Output format:
```json
[
  {
    "id": 0,
    "polygon": [[x1, y1], [x2, y2], ...]
  }
]
```

#### `add_spot(polygon: List[Tuple[int,int]]) → ParkingSpot`

Creates a new spot with auto-incremented ID:
```python
new_id = max((s.spot_id for s in self.spots), default=-1) + 1
```

#### `clear() → None`

Removes all spots from `self.spots`.

---

### Segmentation: `extract_patch(frame, spot) → np.ndarray | None`

**This is the Image Segmentation implementation.**

```python
def extract_patch(self, frame: np.ndarray, spot: ParkingSpot) -> np.ndarray:
    # 1. Create a blank single-channel mask the same size as the frame
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # 2. Fill the polygon with white (255) — Image Segmentation
    cv2.fillPoly(mask, [spot.polygon], 255)

    # 3. Apply mask to frame (pixels outside polygon become black)
    masked = cv2.bitwise_and(frame, frame, mask=mask)

    # 4. Crop to bounding rectangle for efficiency
    x, y, w, h = spot.bbox
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)

    # 5. Guard: return None for degenerate crops (CRITICAL for crash prevention)
    if x2 <= x1 or y2 <= y1:
        return None

    return masked[y1:y2, x1:x2]
```

**CRITICAL:** If the spot polygon has zero or degenerate dimensions in the current frame
(e.g. polygon outside frame bounds), `extract_patch` returns `None`.
All callers of `classify(patch)` must guard for `None` input.

---

### Drawing: `draw_overlays(frame, ...) → np.ndarray`

Draws on a **copy** of `frame` (does not modify in-place).

For each spot:
1. **Semi-transparent fill:**
   ```python
   overlay = out.copy()
   cv2.fillPoly(overlay, [spot.polygon], color)
   cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
   ```
   Alpha = 0.25 (25% fill, 75% original frame)

2. **Polygon outline:**
   ```python
   cv2.polylines(out, [spot.polygon], isClosed=True, color=color, thickness=thickness)
   ```

3. **Spot ID label:**
   ```python
   cv2.putText(out, str(spot.spot_id), (cx-8, cy+5),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 1, cv2.LINE_AA)
   ```

Color selection:
- `"empty"` → `empty_color` (default: `(0, 200, 0)` = green BGR)
- `"occupied"` → `occupied_color` (default: `(0, 0, 220)` = red BGR)
- `"unknown"` and any other value → `unknown_color` (default: `(200, 200, 0)` = yellow BGR)

---

### Statistics Properties

```python
@property
def total(self) -> int:         return len(self.spots)

@property
def empty_count(self) -> int:   return sum(1 for s in self.spots if s.status == "empty")

@property
def occupied_count(self) -> int: return sum(1 for s in self.spots if s.status == "occupied")
```

---

## Imports

```python
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
```

---

## Important: Coordinate System

All polygon coordinates are in **pipeline resolution** (i.e., the resolution after
`VideoHandler._resize()` — default 640px wide).

`draw_rois.py` MUST be used to create `parking_spots.json` because it reads
`process_width` from `config.yaml` and resizes the canvas to the exact same dimensions.

**Never** create polygon coordinates by hand at a different resolution and expect them to work.
