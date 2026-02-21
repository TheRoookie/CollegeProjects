"""
draw_rois.py
------------
Interactive OpenCV tool for defining parking-space ROI polygons.

Controls
--------
  Left-click         Add a vertex to the current polygon.
  Right-click        Complete and save the current polygon as a parking spot.
  'u'                Undo the last added vertex.
  'd'                Delete the last completed spot.
  's'                Save all spots to the JSON file and quit.
  'q' / Escape       Quit without saving.

Usage
-----
  python draw_rois.py --video path/to/video.mp4
  python draw_rois.py --video path/to/video.mp4 --output assets/parking_spots.json
  python draw_rois.py --video path/to/video.mp4 --frame 120   (seek to frame 120)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import yaml


# ---- State -------------------------------------------------------------------

_current_polygon: List[Tuple[int, int]] = []   # vertices being drawn
_spots: List[List[Tuple[int, int]]] = []        # completed polygons
_frame_display: np.ndarray | None = None


def _draw_state(frame: np.ndarray) -> np.ndarray:
    """Render all completed spots + the polygon in progress onto a copy of frame."""
    out = frame.copy()

    # Completed spots
    for i, poly in enumerate(_spots):
        pts = np.array(poly, dtype=np.int32)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], (0, 200, 0))
        cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
        cv2.polylines(out, [pts], isClosed=True, color=(0, 220, 0), thickness=2)
        cx = int(np.mean([p[0] for p in poly]))
        cy = int(np.mean([p[1] for p in poly]))
        cv2.putText(out, str(i), (cx - 6, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # In-progress polygon
    if _current_polygon:
        for pt in _current_polygon:
            cv2.circle(out, pt, 4, (0, 165, 255), -1)
        for j in range(1, len(_current_polygon)):
            cv2.line(out, _current_polygon[j - 1], _current_polygon[j], (0, 165, 255), 1)

    # HUD
    instructions = [
        "Left-click: add vertex",
        "Right-click: complete spot",
        "'u': undo vertex  'd': delete spot",
        "'s': save & quit  'q': quit",
        f"Spots: {len(_spots)}   Vertices: {len(_current_polygon)}",
    ]
    for k, line in enumerate(instructions):
        cv2.putText(out, line, (8, 18 + k * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return out


def _mouse_callback(event: int, x: int, y: int, _flags: int, _param) -> None:
    global _current_polygon
    if event == cv2.EVENT_LBUTTONDOWN:
        _current_polygon.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(_current_polygon) >= 3:
            _spots.append(list(_current_polygon))
            _current_polygon = []
        else:
            print("[draw_rois] Need at least 3 vertices to close a polygon.")


def _load_process_width(config_path: str = "config.yaml") -> int:
    """Read process_width from config.yaml so draw_rois uses the same resolution as the pipeline."""
    try:
        with open(config_path, "r") as fh:
            cfg = yaml.safe_load(fh)
        return int(cfg.get("video", {}).get("process_width", 640))
    except Exception:
        return 640


def run(video_path: str, output_path: str, seek_frame: int = 0, config_path: str = "config.yaml") -> None:
    global _frame_display, _current_polygon

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[draw_rois] Cannot open video: {video_path}")
        sys.exit(1)

    if seek_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, seek_frame)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[draw_rois] Could not read frame from video.")
        sys.exit(1)

    # Resize to EXACTLY the same process_width the pipeline uses.
    # This ensures saved coordinates map 1:1 onto the frames the pipeline processes.
    process_width = _load_process_width(config_path)
    h, w = frame.shape[:2]
    if w != process_width:
        scale = process_width / w
        frame = cv2.resize(frame, (process_width, int(h * scale)), interpolation=cv2.INTER_LINEAR)
    print(f"[draw_rois] Canvas size: {frame.shape[1]}x{frame.shape[0]}  (process_width={process_width})")

    _frame_display = frame.copy()

    window = "Draw Parking ROIs"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, _mouse_callback)

    print("[draw_rois] Window opened. Follow on-screen instructions.")

    while True:
        display = _draw_state(_frame_display)
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("u"):
            if _current_polygon:
                _current_polygon.pop()
        elif key == ord("d"):
            if _spots:
                removed = _spots.pop()
                print(f"[draw_rois] Deleted spot {len(_spots)}: {removed}")
        elif key == ord("s"):
            _save(output_path)
            print(f"[draw_rois] Saved {len(_spots)} spots to {output_path}")
            break
        elif key in (ord("q"), 27):  # 27 = Escape
            print("[draw_rois] Quit without saving.")
            break

    cv2.destroyAllWindows()


def _save(output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = [{"id": i, "polygon": list(poly)} for i, poly in enumerate(_spots)]
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


# ---- CLI entry point ---------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Interactive ROI drawing tool")
    ap.add_argument("--video", required=True, help="Path to the video file")
    ap.add_argument(
        "--output", default="assets/parking_spots.json",
        help="Output JSON file for ROI coordinates"
    )
    ap.add_argument(
        "--frame", type=int, default=0,
        help="Frame index to use as background snapshot"
    )
    ap.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (used to read process_width)"
    )
    args = ap.parse_args()
    run(args.video, args.output, args.frame, args.config)
