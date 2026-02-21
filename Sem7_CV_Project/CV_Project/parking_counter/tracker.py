"""
tracker.py
----------
Centroid Tracker — assigns persistent integer IDs to detected vehicles
across consecutive frames.

Algorithm:
  1. Compute the centroid of each new detection bounding box.
  2. Match each new centroid to the nearest existing tracked centroid
     (within ``max_distance`` pixels) using a greedy nearest-neighbour
     assignment (based on scipy's linear_sum_assignment for optimality).
  3. Unmatched existing tracks have their ``disappeared`` counter incremented.
  4. Tracks that have been missing for more than ``max_disappeared`` frames
     are deregistered.
  5. Unmatched new detections are registered as fresh tracks.

Syllabus topic covered: **Object Tracking**
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import cdist  # type: ignore[import]


@dataclass
class Track:
    """A single tracked vehicle."""

    track_id: int
    centroid: Tuple[int, int]
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    disappeared: int = 0
    history: List[Tuple[int, int]] = field(default_factory=list)


class CentroidTracker:
    """
    Tracks vehicles through a video sequence using centroid matching.

    Parameters
    ----------
    max_distance : int
        Maximum pixel distance within which two centroids can be matched.
    max_disappeared : int
        Number of consecutive frames a track can be absent before removal.
    """

    def __init__(self, max_distance: int = 50, max_disappeared: int = 20) -> None:
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared

        self._next_id: int = 0
        self.tracks: OrderedDict[int, Track] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self, detections: List[Tuple[int, int, int, int]]
    ) -> Dict[int, Track]:
        """
        Update tracks given a list of bounding boxes for the current frame.

        Parameters
        ----------
        detections : list of (x1, y1, x2, y2) tuples
            Bounding boxes of detected vehicles in the current frame.

        Returns
        -------
        dict[int, Track]
            Current active tracks keyed by their integer ID.
        """
        # ---- Case 1: No detections this frame ---
        if len(detections) == 0:
            for track in list(self.tracks.values()):
                track.disappeared += 1
                if track.disappeared > self.max_disappeared:
                    del self.tracks[track.track_id]
            return dict(self.tracks)

        # ---- Step A: Compute input centroids ---------------------
        input_centroids = np.array(
            [_bbox_centroid(b) for b in detections], dtype=np.float32
        )

        # ---- Case 2: No existing tracks → register all ----------
        if len(self.tracks) == 0:
            for bbox, centroid in zip(detections, input_centroids):
                self._register(tuple(centroid.astype(int)), tuple(bbox))  # type: ignore[arg-type]
            return dict(self.tracks)

        # ---- Step B: Build cost matrix & solve assignment -------
        existing_ids = list(self.tracks.keys())
        existing_centroids = np.array(
            [self.tracks[tid].centroid for tid in existing_ids], dtype=np.float32
        )

        cost_matrix = cdist(existing_centroids, input_centroids)

        # Greedy row/col assignment (optimal via scipy)
        from scipy.optimize import linear_sum_assignment  # type: ignore[import]
        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        matched_rows: set[int] = set()
        matched_cols: set[int] = set()

        for row, col in zip(row_idx, col_idx):
            if cost_matrix[row, col] > self.max_distance:
                continue  # Too far apart – treat as separate objects

            tid = existing_ids[row]
            cx, cy = int(input_centroids[col][0]), int(input_centroids[col][1])
            self.tracks[tid].centroid = (cx, cy)
            self.tracks[tid].bbox = detections[col]
            self.tracks[tid].disappeared = 0
            self.tracks[tid].history.append((cx, cy))
            if len(self.tracks[tid].history) > 50:
                self.tracks[tid].history.pop(0)

            matched_rows.add(row)
            matched_cols.add(col)

        # ---- Step C: Handle unmatched existing tracks -----------
        unmatched_rows = set(range(len(existing_ids))) - matched_rows
        for row in unmatched_rows:
            tid = existing_ids[row]
            self.tracks[tid].disappeared += 1
            if self.tracks[tid].disappeared > self.max_disappeared:
                del self.tracks[tid]

        # ---- Step D: Register new detections --------------------
        unmatched_cols = set(range(len(detections))) - matched_cols
        for col in unmatched_cols:
            cx, cy = int(input_centroids[col][0]), int(input_centroids[col][1])
            self._register((cx, cy), detections[col])

        return dict(self.tracks)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _register(
        self,
        centroid: Tuple[int, int],
        bbox: Tuple[int, int, int, int],
    ) -> None:
        """Create a new track."""
        track = Track(
            track_id=self._next_id,
            centroid=centroid,
            bbox=bbox,
            history=[centroid],
        )
        self.tracks[self._next_id] = track
        self._next_id += 1


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _bbox_centroid(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Return the (cx, cy) centroid of a bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
