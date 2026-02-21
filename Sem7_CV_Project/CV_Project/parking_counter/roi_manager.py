"""
roi_manager.py
--------------
Manages parking-space Regions of Interest (ROIs).

Each parking space is represented as a polygon (list of (x, y) vertices).
The module handles:
  * Loading / saving ROI coordinates from / to a JSON file.
  * Extracting a masked patch for each ROI from a frame (segmentation).
  * Drawing ROI overlays on a display frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ParkingSpot:
    """Represents a single parking space polygon."""

    spot_id: int
    polygon: np.ndarray          # shape (N, 2), dtype int32
    status: str = "unknown"      # "empty" | "occupied" | "unknown"

    # Bounding-box cache (computed lazily)
    _bbox: Optional[Tuple[int, int, int, int]] = field(default=None, repr=False)

    # Debounce state — internal, not part of public repr
    _candidate: str = field(default="unknown", repr=False)
    _candidate_count: int = field(default=0, repr=False)

    def update_status(self, new_status: str, debounce: int = 3) -> None:
        """
        Update the spot status only after *debounce* consecutive frames
        agree on *new_status*.  This prevents single-frame flicker and
        ensures a departed car is only marked empty once several frames
        in a row confirm the space is clear.
        """
        if new_status == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = new_status
            self._candidate_count = 1

        if self._candidate_count >= debounce:
            self.status = self._candidate

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Axis-aligned bounding box (x, y, w, h) of the polygon."""
        if self._bbox is None:
            x, y, w, h = cv2.boundingRect(self.polygon)
            self._bbox = (x, y, w, h)
        return self._bbox

    @property
    def center(self) -> Tuple[int, int]:
        """Centroid of the bounding box."""
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)


class ROIManager:
    """
    Loads, stores, and applies parking-space ROIs.

    Parameters
    ----------
    coordinates_file : str | Path
        Path to the JSON file with polygon coordinates.
        Format expected::

            [
              {"id": 0, "polygon": [[x1,y1],[x2,y2],...]},
              ...
            ]
    """

    def __init__(self, coordinates_file: str | Path) -> None:
        self.coordinates_file = Path(coordinates_file)
        self.spots: List[ParkingSpot] = []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """
        Load ROIs from ``coordinates_file``.

        Returns ``True`` on success, ``False`` if the file is missing.
        """
        if not self.coordinates_file.exists():
            return False

        with open(self.coordinates_file, "r", encoding="utf-8") as fh:
            data: List[Dict] = json.load(fh)

        self.spots = [
            ParkingSpot(
                spot_id=entry["id"],
                polygon=np.array(entry["polygon"], dtype=np.int32),
            )
            for entry in data
        ]
        return True

    def save(self) -> None:
        """Persist current ROIs to ``coordinates_file``."""
        self.coordinates_file.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {"id": s.spot_id, "polygon": s.polygon.tolist()}
            for s in self.spots
        ]
        with open(self.coordinates_file, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def add_spot(self, polygon: List[Tuple[int, int]]) -> ParkingSpot:
        """Add a new parking spot and return it."""
        new_id = max((s.spot_id for s in self.spots), default=-1) + 1
        spot = ParkingSpot(
            spot_id=new_id,
            polygon=np.array(polygon, dtype=np.int32),
        )
        self.spots.append(spot)
        return spot

    def clear(self) -> None:
        """Remove all spots."""
        self.spots.clear()

    # ------------------------------------------------------------------
    # Segmentation helpers
    # ------------------------------------------------------------------

    def extract_patch(
        self, frame: np.ndarray, spot: ParkingSpot
    ) -> np.ndarray:
        """
        Extract a masked RGB patch for *spot* from *frame*.

        Uses ``cv2.fillPoly`` + ``cv2.bitwise_and`` to isolate the polygon
        region, then crops to the bounding box for downstream processing.

        Returns
        -------
        np.ndarray
            Cropped BGR patch with pixels outside the polygon zeroed out.
        """
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [spot.polygon], 255)           # --- Segmentation ---
        masked = cv2.bitwise_and(frame, frame, mask=mask) # --- Masking ---

        # Crop to bounding rect for efficiency
        x, y, w, h = spot.bbox
        # Clamp to frame bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)

        # Guard: return None if the crop is degenerate (zero size in any axis)
        if x2 <= x1 or y2 <= y1:
            return None  # type: ignore[return-value]

        return masked[y1:y2, x1:x2]

    # ------------------------------------------------------------------
    # Overlay drawing
    # ------------------------------------------------------------------

    def draw_overlays(
        self,
        frame: np.ndarray,
        empty_color: Tuple[int, int, int] = (0, 200, 0),
        occupied_color: Tuple[int, int, int] = (0, 0, 220),
        unknown_color: Tuple[int, int, int] = (200, 200, 0),
        thickness: int = 2,
        font_scale: float = 0.5,
    ) -> np.ndarray:
        """
        Draw polygon outlines and status labels for every spot on *frame*.

        Returns a copy of the frame with overlays applied.
        """
        out = frame.copy()
        color_map = {
            "empty":    empty_color,
            "occupied": occupied_color,
            "unknown":  unknown_color,
        }
        for spot in self.spots:
            color = color_map.get(spot.status, unknown_color)
            # Draw semi-transparent fill
            overlay = out.copy()
            cv2.fillPoly(overlay, [spot.polygon], color)
            cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
            # Draw polygon outline
            cv2.polylines(out, [spot.polygon], isClosed=True, color=color, thickness=thickness)
            # Label
            cx, cy = spot.center
            cv2.putText(
                out,
                str(spot.spot_id),
                (cx - 8, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return out

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def total(self) -> int:
        return len(self.spots)

    @property
    def empty_count(self) -> int:
        return sum(1 for s in self.spots if s.status == "empty")

    @property
    def occupied_count(self) -> int:
        return sum(1 for s in self.spots if s.status == "occupied")
