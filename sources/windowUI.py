import json
import time
from datetime import datetime
from pathlib import Path

import cv2

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QStyle,
    QMessageBox,
    QScrollArea,
    QComboBox,
    QSplitter,
)


SPECIES_OPTIONS = [
    "unknown",
    "species_a",
    "species_b",
    "species_c",
    "species_d",
]


class DetectionItemWidget(QWidget):
    """Widget showing one saved crop and an editable label."""

    def __init__(self, item_data, species_options):
        super().__init__()
        self.item_data = item_data

        layout = QHBoxLayout(self)

        self.crop_label = QLabel()
        self.crop_label.setFixedSize(100, 100)
        self.crop_label.setAlignment(Qt.AlignCenter)

        crop_path = Path(self.item_data["crop_file"])
        pixmap = QPixmap(str(crop_path))
        if not pixmap.isNull():
            self.crop_label.setPixmap(
                pixmap.scaled(
                    self.crop_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        else:
            self.crop_label.setText("Image\nnot found")

        layout.addWidget(self.crop_label)

        info_layout = QVBoxLayout()

        self.id_label = QLabel(f"ID: {self.item_data['id']}")
        self.det_label = QLabel(
            f"Detected: {self.item_data['detected_label']} ({self.item_data['confidence']:.2f})"
        )

        self.combo = QComboBox()
        self.combo.addItems(species_options)

        current_label = self.item_data.get("editable_label", "unknown")
        index = self.combo.findText(current_label)
        if index >= 0:
            self.combo.setCurrentIndex(index)

        info_layout.addWidget(self.id_label)
        info_layout.addWidget(self.det_label)
        info_layout.addWidget(self.combo)

        layout.addLayout(info_layout)

    def get_selected_label(self):
        return self.combo.currentText()


class VideoWindow(QMainWindow):
    """Main Qt window displaying video frames, save tools, and review sidebar."""

    def __init__(self, source):
        super().__init__()

        self.source = source
        self.current_pixmap = None
        self.is_paused = False
        self.interval_ms = 33

        # Current inference data
        self.last_raw_frame = None
        self.last_display_frame = None
        self.last_detections = []
        self.saved_frame_index = 0

        # Dataset paths
        self.dataset_root = Path("dataset")
        self.crops_dir = self.dataset_root / "crops"
        self.frames_dir = self.dataset_root / "frames"
        self.metadata_path = self.dataset_root / "metadata.json"

        # FPS tracking
        self.frame_count = 0
        self.last_fps_timestamp = time.time()
        self.current_fps = 0.0

        # Review panel widgets
        self.review_item_widgets = []

        self.setWindowTitle("Foram detection")
        self.resize(1280, 760)

        self._build_ui()
        self._ensure_dataset_dirs()
        self.load_review_items()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self._init_source()

    def _build_ui(self):
        """Create the UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)

        splitter = QSplitter(Qt.Horizontal)

        # Left side: video + controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.video_label = QLabel("Initializing video...")
        self.video_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.video_label, stretch=1)

        controls_layout = QHBoxLayout()

        self.play_pause_button = QPushButton()
        self.play_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.play_pause_button.clicked.connect(self.toggle_pause)

        self.save_button = QPushButton("Save detections")
        self.save_button.clicked.connect(self.save_current_detections)

        self.refresh_review_button = QPushButton("Refresh review")
        self.refresh_review_button.clicked.connect(self.load_review_items)

        controls_layout.addWidget(self.play_pause_button)
        controls_layout.addWidget(self.save_button)
        controls_layout.addWidget(self.refresh_review_button)

        self.status_label = QLabel("Status: Idle")
        controls_layout.addWidget(self.status_label)

        controls_layout.addStretch()

        self.fps_label = QLabel("FPS: 0.0")
        controls_layout.addWidget(self.fps_label)

        left_layout.addLayout(controls_layout)

        # Right side: review panel
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.review_title_label = QLabel("Saved detections")
        right_layout.addWidget(self.review_title_label)

        self.review_scroll_area = QScrollArea()
        self.review_scroll_area.setWidgetResizable(True)
        right_layout.addWidget(self.review_scroll_area, stretch=1)

        self.review_scroll_container = QWidget()
        self.review_scroll_layout = QVBoxLayout(self.review_scroll_container)
        self.review_scroll_layout.setAlignment(Qt.AlignTop)
        self.review_scroll_area.setWidget(self.review_scroll_container)

        self.save_labels_button = QPushButton("Save label corrections")
        self.save_labels_button.clicked.connect(self.save_review_labels)
        right_layout.addWidget(self.save_labels_button)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([900, 380])

        root_layout.addWidget(splitter)

    def _ensure_dataset_dirs(self):
        """Create dataset folders if they do not exist."""
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        if not self.metadata_path.exists():
            self._write_metadata([])

    def _read_metadata(self):
        if not self.metadata_path.exists():
            return []

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_metadata(self, data):
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _init_source(self):
        """Open the video source and start the update timer."""
        if not self.source.open():
            self.video_label.setText("Failed to open video source")
            self.status_label.setText("Status: Error")
            self.play_pause_button.setEnabled(False)
            self.save_button.setEnabled(False)
            return

        fps = self.source.get_fps()
        self.interval_ms = max(1, int(1000 / fps))

        self.status_label.setText("Status: Playing")
        self.timer.start(self.interval_ms)

    def toggle_pause(self):
        """Toggle between paused and playing states."""
        if self.is_paused:
            self.timer.start(self.interval_ms)
            self.is_paused = False
            self.play_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.status_label.setText("Status: Playing")
        else:
            self.timer.stop()
            self.is_paused = True
            self.play_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.status_label.setText("Status: Paused")

    def update_frame(self):
        """Fetch a frame from the source and update the display."""
        ret, payload = self.source.read()

        if not ret or payload is None:
            self.timer.stop()
            self.status_label.setText("Status: End / Error")
            self.video_label.setText("End of stream or read error")
            self.play_pause_button.setEnabled(False)
            return

        self.last_raw_frame = payload["raw_frame"]
        self.last_display_frame = payload["display_frame"]
        self.last_detections = payload["detections"]

        self.current_pixmap = self.frame_to_pixmap(self.last_display_frame)
        self.refresh_display()
        self.update_fps()

    def frame_to_pixmap(self, frame):
        """Convert an OpenCV BGR frame to QPixmap."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        height, width, channels = frame_rgb.shape
        bytes_per_line = channels * width

        image = QImage(
            frame_rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        )

        return QPixmap.fromImage(image)

    def refresh_display(self):
        """Scale and display the current pixmap."""
        if self.current_pixmap is None:
            return

        scaled = self.current_pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def update_fps(self):
        """Compute and refresh the displayed FPS."""
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.last_fps_timestamp

        if elapsed >= 1.0:
            self.current_fps = self.frame_count / elapsed
            self.fps_label.setText(f"FPS: {self.current_fps:.1f}")
            self.frame_count = 0
            self.last_fps_timestamp = now

    def save_current_detections(self):
        """
        Save crops and metadata only when the user clicks the button.
        Nothing is saved automatically.
        """
        if self.last_raw_frame is None:
            QMessageBox.warning(self, "Save detections", "No frame available yet.")
            return

        if not self.last_detections:
            QMessageBox.information(self, "Save detections", "No detections on current frame.")
            return

        self.saved_frame_index += 1
        frame_id = f"frame_{self.saved_frame_index:06d}"
        timestamp = datetime.now().isoformat(timespec="seconds")

        annotated_frame_path = self.frames_dir / f"{frame_id}_annotated.png"
        cv2.imwrite(str(annotated_frame_path), self.last_display_frame)

        metadata = self._read_metadata()
        saved_count = 0

        frame_h, frame_w = self.last_raw_frame.shape[:2]

        for det_index, det in enumerate(self.last_detections):
            x1, y1, x2, y2 = det["bbox"]

            x1 = max(0, min(x1, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            x2 = max(0, min(x2, frame_w))
            y2 = max(0, min(y2, frame_h))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = self.last_raw_frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_filename = f"{frame_id}_det_{det_index:03d}.png"
            crop_path = self.crops_dir / crop_filename
            cv2.imwrite(str(crop_path), crop)

            metadata.append(
                {
                    "id": f"{frame_id}_det_{det_index:03d}",
                    "timestamp": timestamp,
                    "frame_id": frame_id,
                    "crop_file": str(crop_path.as_posix()),
                    "annotated_frame_file": str(annotated_frame_path.as_posix()),
                    "bbox": [x1, y1, x2, y2],
                    "detected_label": det["detected_label"],
                    "confidence": det["confidence"],
                    "editable_label": det["editable_label"],
                }
            )
            saved_count += 1

        self._write_metadata(metadata)
        self.load_review_items()

        QMessageBox.information(
            self,
            "Save detections",
            f"Saved {saved_count} crop(s) for {frame_id}.",
        )

    def load_review_items(self):
        """Reload the review sidebar from metadata."""
        metadata = self._read_metadata()

        while self.review_scroll_layout.count():
            child = self.review_scroll_layout.takeAt(0)
            widget = child.widget()
            if widget is not None:
                widget.deleteLater()

        self.review_item_widgets = []

        if not metadata:
            self.review_title_label.setText("Saved detections (0)")
            return

        self.review_title_label.setText(f"Saved detections ({len(metadata)})")

        for item_data in metadata:
            item_widget = DetectionItemWidget(item_data, SPECIES_OPTIONS)
            self.review_scroll_layout.addWidget(item_widget)
            self.review_item_widgets.append(item_widget)

    def save_review_labels(self):
        """Save manual label corrections from the right sidebar."""
        metadata = self._read_metadata()

        if not metadata:
            QMessageBox.information(self, "Review", "No saved detections to update.")
            return

        for item_data, item_widget in zip(metadata, self.review_item_widgets):
            item_data["editable_label"] = item_widget.get_selected_label()

        self._write_metadata(metadata)
        QMessageBox.information(self, "Review", "Label corrections saved.")

    def resizeEvent(self, event):
        """Handle window resize events."""
        super().resizeEvent(event)
        self.refresh_display()

    def closeEvent(self, event):
        """Release resources when closing the window."""
        self.timer.stop()
        self.source.release()
        super().closeEvent(event)