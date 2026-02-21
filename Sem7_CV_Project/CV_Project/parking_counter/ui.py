"""
ui.py
-----
PyQt5 main window for the Car Parking Space Counter.

Layout
------
┌──────────────────────────────────────────────────────────┐
│  Toolbar: [Open Video] [Load ROIs] [Draw ROIs] [Start/Stop]│
├──────────────────┬───────────────────────────────────────┤
│  Video Display   │         Analytics Sidebar             │
│  (live feed)     │  Total:    [ n ]                      │
│                  │  Empty:    [ n ]  ██████  (green)      │
│                  │  Occupied: [ n ]  ██████  (red)        │
│                  │                                        │
│                  │  [Progress bar — occupancy %]          │
│                  │                                        │
│                  │  Status log                            │
└──────────────────┴───────────────────────────────────────┘
│  Status bar                                              │
└──────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from classifier import SpaceClassifier
from detector import VehicleDetector
from pipeline import PipelineThread
from roi_manager import ROIManager
from tracker import CentroidTracker
from video_handler import VideoHandler


# ============================================================
# Video display widget
# ============================================================

class VideoLabel(QLabel):
    """QLabel that scales a QPixmap while preserving aspect ratio."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 270)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #1a1a2e; border-radius: 8px;")
        self._pixmap: Optional[QPixmap] = None
        # Placeholder text
        self.setText("No video loaded")
        self.setFont(QFont("Segoe UI", 14))
        self.setStyleSheet(
            "QLabel { background-color: #1a1a2e; color: #555; border-radius: 8px; }"
        )

    def set_frame(self, frame: np.ndarray) -> None:
        """Convert a BGR numpy frame to a QPixmap and display it."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._refresh()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(scaled)


# ============================================================
# Sidebar analytics widget
# ============================================================

class AnalyticsSidebar(QWidget):
    """Shows live parking counts and occupancy bar."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        title = QLabel("Analytics")
        title.setFont(QFont("Segoe UI", 13, QFont.Bold))
        title.setStyleSheet("color: #e0e0e0;")
        layout.addWidget(title)

        self._lbl_total = self._stat_label("Total Spaces", "#e0e0e0")
        self._lbl_empty = self._stat_label("Available", "#00c853")
        self._lbl_occupied = self._stat_label("Occupied", "#ff1744")

        for lbl in (self._lbl_total, self._lbl_empty, self._lbl_occupied):
            layout.addWidget(lbl)

        layout.addSpacing(8)
        bar_lbl = QLabel("Occupancy")
        bar_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(bar_lbl)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._bar.setStyleSheet(
            "QProgressBar { background: #333; border-radius: 6px; height: 14px; }"
            "QProgressBar::chunk { background: #ff1744; border-radius: 6px; }"
        )
        layout.addWidget(self._bar)

        layout.addSpacing(10)
        log_title = QLabel("Event Log")
        log_title.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(log_title)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(200)
        self._log.setStyleSheet(
            "background: #12122a; color: #9e9e9e; border-radius: 4px; font-size: 10px;"
        )
        layout.addWidget(self._log)

        layout.addStretch()

    # ------------------------------------------------------------------

    def update_stats(self, total: int, empty: int, occupied: int) -> None:
        self._lbl_total.setText(f"Total Spaces:  {total}")
        self._lbl_empty.setText(f"Available:     {empty}")
        self._lbl_occupied.setText(f"Occupied:      {occupied}")
        pct = int(occupied / total * 100) if total > 0 else 0
        self._bar.setValue(pct)
        self._bar.setFormat(f"{pct}%")

    def log(self, msg: str) -> None:
        self._log.append(msg)

    # ------------------------------------------------------------------

    @staticmethod
    def _stat_label(text: str, color: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {color}; font-size: 13px; padding: 4px 6px;"
            "background: #16213e; border-radius: 4px;"
        )
        lbl.setFont(QFont("Consolas", 11))
        return lbl


# ============================================================
# Main Window
# ============================================================

class MainWindow(QMainWindow):
    """Application main window."""

    TITLE = "Car Parking Space Counter"

    def __init__(self, config_path: str = "config.yaml") -> None:
        super().__init__()

        # Load configuration
        with open(config_path, "r") as fh:
            self._cfg = yaml.safe_load(fh)

        self._pipeline_thread: Optional[PipelineThread] = None
        self._video_source: Optional[str] = None

        self._build_ui()
        self._apply_dark_theme()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.setWindowTitle(self.TITLE)
        self.resize(1100, 650)

        # ---- Toolbar ------------------------------------------------
        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setMovable(False)
        toolbar.setStyleSheet("QToolBar { spacing: 6px; padding: 4px; }")
        self.addToolBar(toolbar)

        self._btn_open = self._tb_button("Open Video", toolbar, self._on_open_video)
        self._btn_roi_load = self._tb_button("Load ROIs", toolbar, self._on_load_rois)
        self._btn_roi_draw = self._tb_button("Draw ROIs", toolbar, self._on_draw_rois)
        toolbar.addSeparator()
        self._btn_start = self._tb_button("▶  Start", toolbar, self._on_start_stop)
        self._btn_start.setStyleSheet(
            "QPushButton { background: #00897b; color: white; border-radius: 4px;"
            " padding: 4px 12px; font-weight: bold; }"
            "QPushButton:hover { background: #00695c; }"
        )

        # ---- Central layout -----------------------------------------
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Video display
        self._video_label = VideoLabel()
        main_layout.addWidget(self._video_label, stretch=4)

        # Sidebar
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setStyleSheet("color: #333;")
        main_layout.addWidget(separator)

        self._sidebar = AnalyticsSidebar()
        main_layout.addWidget(self._sidebar, stretch=0)

        # ---- Status bar ---------------------------------------------
        self._status_bar = QStatusBar()
        self._status_bar.showMessage("Ready — open a video file to begin.")
        self.setStatusBar(self._status_bar)

    # ------------------------------------------------------------------
    # Button actions
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_open_video(self) -> None:
        """Open a file dialog to select a video file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov *.webm);;All Files (*)"
        )
        if path:
            self._video_source = path
            self._status_bar.showMessage(f"Video loaded: {Path(path).name}")
            self._sidebar.log(f"[Video] Loaded: {Path(path).name}")

    @pyqtSlot()
    def _on_load_rois(self) -> None:
        """Load parking-space polygon coordinates from a JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load ROI File", "assets/", "JSON Files (*.json);;All Files (*)"
        )
        if path:
            roi_mgr = self._get_or_create_roi_manager()
            roi_mgr.coordinates_file = Path(path)
            success = roi_mgr.load()
            if success:
                msg = f"Loaded {roi_mgr.total} parking spots from {Path(path).name}"
                self._status_bar.showMessage(msg)
                self._sidebar.log(f"[ROI] {msg}")
            else:
                QMessageBox.warning(self, "ROI Load Error", f"Could not load: {path}")

    @pyqtSlot()
    def _on_draw_rois(self) -> None:
        """Launch the interactive ROI drawing tool in a subprocess."""
        if not self._video_source:
            QMessageBox.information(self, "No Video", "Please open a video file first.")
            return
        import subprocess, sys
        cwd = str(Path(__file__).parent)
        subprocess.Popen(
            [
                sys.executable, "draw_rois.py",
                "--video", self._video_source,
                "--output", self._cfg["roi"]["coordinates_file"],
                "--config", "config.yaml",
            ],
            cwd=cwd,
        )
        self._sidebar.log("[ROI] ROI drawing tool launched.")

    @pyqtSlot()
    def _on_start_stop(self) -> None:
        """Toggle the processing pipeline on / off."""
        if self._pipeline_thread and self._pipeline_thread.isRunning():
            self._stop_pipeline()
        else:
            self._start_pipeline()

    # ------------------------------------------------------------------
    # Pipeline management
    # ------------------------------------------------------------------

    def _start_pipeline(self) -> None:
        if not self._video_source:
            QMessageBox.information(self, "No Video", "Please open a video file first.")
            return

        roi_mgr = self._get_or_create_roi_manager()
        if roi_mgr.total == 0:
            reply = QMessageBox.question(
                self, "No ROIs defined",
                "No parking-space ROIs are loaded.\n"
                "The system will run detection & tracking without space classification.\n"
                "Continue anyway?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        vcfg = self._cfg["video"]
        det_cfg = self._cfg["detection"]
        trk_cfg = self._cfg["tracker"]
        cls_cfg = self._cfg["classifier"]
        disp_cfg = self._cfg["display"]

        video_handler = VideoHandler(
            source=self._video_source,
            process_width=vcfg.get("process_width", 640),
            max_fps=vcfg.get("max_fps", 30),
        )
        detector = VehicleDetector(
            model_path=det_cfg.get("model_weights", "assets/yolov8n.pt"),
            confidence=det_cfg.get("confidence_threshold", 0.45),
            iou_threshold=det_cfg.get("iou_threshold", 0.45),
            vehicle_classes=det_cfg.get("vehicle_classes", [2, 3, 5, 7]),
            device=det_cfg.get("device", "cpu"),
        )
        tracker = CentroidTracker(
            max_distance=trk_cfg.get("max_distance", 50),
            max_disappeared=trk_cfg.get("max_disappeared", 20),
        )
        method = cls_cfg.get("method", "overlap")
        if method == "cnn":
            # CNN backend — only used when explicitly requested and a checkpoint exists
            classifier = SpaceClassifier(method="cnn", checkpoint_path=cls_cfg.get("cnn_checkpoint", ""))
        else:
            # "overlap" and "pixel_count" both use pixel_count as the SpaceClassifier
            # backend. For "overlap", this is only called as a fallback when YOLO fails.
            classifier = SpaceClassifier(
                method="pixel_count",
                count_threshold=cls_cfg.get("count_threshold", 900),
                blur_kernel=cls_cfg.get("blur_kernel", 3),
                adaptive_block=cls_cfg.get("adaptive_block", 25),
                adaptive_c=cls_cfg.get("adaptive_c", 16),
            )

        use_overlap = cls_cfg.get("method", "overlap") == "overlap"
        self._pipeline_thread = PipelineThread(
            video_handler=video_handler,
            roi_manager=roi_mgr,
            detector=detector,
            tracker=tracker,
            classifier=classifier,
            display_cfg=disp_cfg,
            detection_fps=vcfg.get("detection_fps", 1.0),
            status_debounce=cls_cfg.get("status_debounce", 2),
            use_detection_overlap=use_overlap,
            overlap_threshold=cls_cfg.get("overlap_threshold", 0.25),
            parent=self,
        )
        self._pipeline_thread.frame_ready.connect(self._on_frame_ready)
        self._pipeline_thread.stats_updated.connect(self._on_stats_updated)
        self._pipeline_thread.error_occurred.connect(self._on_pipeline_error)
        self._pipeline_thread.finished_signal.connect(self._on_pipeline_finished)

        self._pipeline_thread.start()

        self._btn_start.setText("■  Stop")
        self._btn_start.setStyleSheet(
            "QPushButton { background: #c62828; color: white; border-radius: 4px;"
            " padding: 4px 12px; font-weight: bold; }"
            "QPushButton:hover { background: #b71c1c; }"
        )
        self._status_bar.showMessage("Processing…")
        self._sidebar.log("[Pipeline] Started.")

    def _stop_pipeline(self) -> None:
        if self._pipeline_thread:
            self._pipeline_thread.stop()
        self._btn_start.setText("▶  Start")
        self._btn_start.setStyleSheet(
            "QPushButton { background: #00897b; color: white; border-radius: 4px;"
            " padding: 4px 12px; font-weight: bold; }"
            "QPushButton:hover { background: #00695c; }"
        )
        self._status_bar.showMessage("Stopped.")
        self._sidebar.log("[Pipeline] Stopped.")

    # ------------------------------------------------------------------
    # Slots connected to the pipeline thread
    # ------------------------------------------------------------------

    @pyqtSlot(np.ndarray)
    def _on_frame_ready(self, frame: np.ndarray) -> None:
        self._video_label.set_frame(frame)

    @pyqtSlot(int, int, int)
    def _on_stats_updated(self, total: int, empty: int, occupied: int) -> None:
        self._sidebar.update_stats(total, empty, occupied)

    @pyqtSlot(str)
    def _on_pipeline_error(self, message: str) -> None:
        self._sidebar.log(f"[Error] {message}")
        self._status_bar.showMessage(f"Error: {message}")

    @pyqtSlot()
    def _on_pipeline_finished(self) -> None:
        self._stop_pipeline()
        self._sidebar.log("[Pipeline] Finished (end of stream).")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_or_create_roi_manager(self) -> ROIManager:
        """Retrieve the ROI manager, creating it fresh if needed."""
        if not hasattr(self, "_roi_manager"):
            roi_file = self._cfg["roi"]["coordinates_file"]
            self._roi_manager = ROIManager(roi_file)
            self._roi_manager.load()  # safe — returns False if missing
        return self._roi_manager

    @staticmethod
    def _tb_button(label: str, toolbar: QToolBar, slot) -> QPushButton:
        btn = QPushButton(label)
        btn.setStyleSheet(
            "QPushButton { background: #2a2a4a; color: #ddd; border-radius: 4px;"
            " padding: 4px 10px; }"
            "QPushButton:hover { background: #3a3a6a; }"
        )
        btn.clicked.connect(slot)
        toolbar.addWidget(btn)
        return btn

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            "QMainWindow { background-color: #12121f; }"
            "QToolBar { background-color: #1e1e3a; border-bottom: 1px solid #333; }"
            "QStatusBar { background-color: #1e1e3a; color: #aaa; font-size: 11px; }"
        )

    def closeEvent(self, event) -> None:
        if self._pipeline_thread and self._pipeline_thread.isRunning():
            self._pipeline_thread.stop()
            self._pipeline_thread.wait(3000)
        event.accept()
