# 08 — Module: `tracker.py`

**Syllabus Topic:** Object Tracking
**File:** `parking_counter/tracker.py` (200 lines)

---

## Purpose

Implements a **centroid-based multi-object tracker** that:
1. Assigns a persistent integer ID to each detected vehicle
2. Maintains ID continuity across frames even when detections momentarily disappear
3. Keeps a position history trail for each track (for visualisation)
4. Uses the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`) for optimal matching

---

## Dataclass: `Track`

```python
@dataclass
class Track:
    track_id: int
    centroid: Tuple[int, int]             # Current (cx, cy) centroid
    bbox: Tuple[int, int, int, int]       # Current (x1, y1, x2, y2) bounding box
    disappeared: int = 0                  # Consecutive frames without a matching detection
    history: List[Tuple[int, int]] = field(default_factory=list)  # Position trail
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `track_id` | int | Unique, monotonically increasing ID (never reused) |
| `centroid` | (int, int) | Centre pixel of the current bounding box |
| `bbox` | (x1,y1,x2,y2) | Most recent bounding box |
| `disappeared` | int | Incremented each frame this track has no matching detection |
| `history` | list | List of past centroid positions; capped at 50 entries |

---

## Class: `CentroidTracker`

### Constructor

```python
CentroidTracker(
    max_distance: int = 50,
    max_disappeared: int = 20,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_distance` | 50 | Maximum pixel distance for a valid centroid match. Pairs farther apart are treated as different vehicles. |
| `max_disappeared` | 20 | If a track is absent for this many consecutive frames, it is permanently removed. |

**Internal state:**
- `self._next_id: int = 0` — next ID to assign
- `self.tracks: OrderedDict[int, Track] = OrderedDict()` — active tracks keyed by ID

---

### Method: `update(detections: List[Tuple[int,int,int,int]]) → Dict[int, Track]`

The core tracking update. Called every detection frame.

**Signature:**
```python
def update(self, detections: List[Tuple[int,int,int,int]]) -> Dict[int, Track]:
```
`detections` is a list of `(x1, y1, x2, y2)` bounding boxes (not `Detection` objects).

**Step-by-step algorithm:**

#### Case 1: No detections this frame
```python
if len(detections) == 0:
    for track in list(self.tracks.values()):
        track.disappeared += 1
        if track.disappeared > self.max_disappeared:
            del self.tracks[track.track_id]
    return dict(self.tracks)
```
All existing tracks are aged. Stale tracks are removed.

#### Case 2: No existing tracks
```python
if len(self.tracks) == 0:
    for bbox, centroid in zip(detections, input_centroids):
        self._register(tuple(centroid.astype(int)), tuple(bbox))
    return dict(self.tracks)
```
All detections become new tracks immediately.

#### General case: Existing tracks AND new detections

**Step A — Compute centroids for all new detections:**
```python
input_centroids = np.array(
    [_bbox_centroid(b) for b in detections], dtype=np.float32
)
```

**Step B — Build cost matrix:**
```python
existing_centroids = np.array(
    [self.tracks[tid].centroid for tid in existing_ids], dtype=np.float32
)
cost_matrix = cdist(existing_centroids, input_centroids)
# Shape: (num_existing_tracks, num_new_detections)
# cost_matrix[i][j] = Euclidean distance between existing track i and detection j
```

**Step C — Solve optimal assignment (Hungarian algorithm):**
```python
from scipy.optimize import linear_sum_assignment
row_idx, col_idx = linear_sum_assignment(cost_matrix)
```
`linear_sum_assignment` minimises the total cost (sum of matched distances).

**Step D — Apply matches (only if distance ≤ max_distance):**
```python
for row, col in zip(row_idx, col_idx):
    if cost_matrix[row, col] > self.max_distance:
        continue  # Too far apart — do NOT match

    tid = existing_ids[row]
    cx, cy = int(input_centroids[col][0]), int(input_centroids[col][1])

    self.tracks[tid].centroid = (cx, cy)
    self.tracks[tid].bbox = detections[col]
    self.tracks[tid].disappeared = 0        # reset counter
    self.tracks[tid].history.append((cx, cy))
    if len(self.tracks[tid].history) > 50:
        self.tracks[tid].history.pop(0)     # cap history at 50 points

    matched_rows.add(row)
    matched_cols.add(col)
```

**Step E — Age unmatched existing tracks:**
```python
unmatched_rows = set(range(len(existing_ids))) - matched_rows
for row in unmatched_rows:
    tid = existing_ids[row]
    self.tracks[tid].disappeared += 1
    if self.tracks[tid].disappeared > self.max_disappeared:
        del self.tracks[tid]
```

**Step F — Register new unmatched detections:**
```python
unmatched_cols = set(range(len(detections))) - matched_cols
for col in unmatched_cols:
    cx, cy = int(input_centroids[col][0]), int(input_centroids[col][1])
    self._register((cx, cy), detections[col])
```

**Returns:** `dict(self.tracks)` — a snapshot of all currently active tracks.

---

### Private Method: `_register(centroid, bbox) → None`

```python
def _register(self, centroid: Tuple[int,int], bbox: Tuple[int,int,int,int]) -> None:
    track = Track(
        track_id=self._next_id,
        centroid=centroid,
        bbox=bbox,
        history=[centroid],
    )
    self.tracks[self._next_id] = track
    self._next_id += 1
```

IDs are never reused. A new vehicle always gets a higher ID than any previous vehicle.

---

## Module-Level Helper: `_bbox_centroid`

```python
def _bbox_centroid(bbox: Tuple[int,int,int,int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
```

---

## Imports

```python
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
from scipy.spatial.distance import cdist
# scipy.optimize.linear_sum_assignment is imported lazily inside update()
```

---

## How `pipeline.py` Uses This Module

```python
# In _process_frame():
bboxes = [d.bbox for d in detections]  # Extract just the bbox tuples
self._tracker.update(bboxes)           # Returns updated tracks dict

# Drawing tracker trails:
for track in self._tracker.tracks.values():
    if len(track.history) > 1:
        pts = np.array(track.history, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(annotated, [pts], False, (255, 255, 0), 1)  # Yellow
```

Note: The tracker is updated on every detection frame (`run_detection=True`).
On non-detection frames, `_tracker.update()` is NOT called — tracks persist with their
last known positions. This is intentional to avoid aging out tracks between detections.

---

## Algorithm Complexity

| Step | Complexity |
|------|-----------|
| Centroid computation | O(N) where N = num detections |
| `cdist` cost matrix | O(M × N) where M = existing tracks, N = detections |
| `linear_sum_assignment` | O(min(M,N)³) — typically very fast for <50 objects |
| Total per frame | Effectively O(N²) for typical parking lot scenes |

---

## Tuning Guide

| Scenario | Adjustment |
|----------|-----------|
| Vehicle enters frame and is immediately deregistered | Increase `max_disappeared` (allows longer absence tolerance) |
| Two vehicles swap IDs when crossing paths | Decrease `max_distance` (stricter matching) |
| Stationary vehicles get wrong ID after brief occlusion | Decrease `max_distance`, increase `max_disappeared` |
| Debris or shadows get tracked as vehicles | Fix in `detector.py` by raising `confidence_threshold` |
