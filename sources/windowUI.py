import json
import re
import time
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
    QLineEdit,
)


def sanitize_name(value: str) -> str:
    """Sanitize a string so it is safe for folder/file names."""
    value = value.strip()
    value = value.replace(" ", "_")
    value = re.sub(r'[<>:"/\\|?*]+', "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("._")


class ImagePreviewWindow(QMainWindow):
    """Separate window used to display one crop in large size."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crop preview")
        self.resize(700, 700)

        self.image_label = QLabel("No image selected")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.image_label)

        self.current_pixmap = None

    def set_crop(self, crop_bgr):
        """Display a crop in the preview window."""
        if crop_bgr is None or crop_bgr.size == 0:
            self.image_label.setText("Invalid crop")
            self.image_label.setPixmap(QPixmap())
            self.current_pixmap = None
            return

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = crop_rgb.shape
        bytes_per_line = c * w

        image = QImage(
            crop_rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888,
        )

        self.current_pixmap = QPixmap.fromImage(image)
        self._refresh_display()

    def _refresh_display(self):
        if self.current_pixmap is None:
            return

        scaled = self.current_pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_display()

class PreviewDetectionItemWidget(QWidget):
    """Widget showing one current crop and an editable species field."""

    def __init__(
        self,
        crop_bgr,
        detection_data,
        species_options,
        on_crop_clicked=None,
        on_remove_clicked=None,
    ):
        super().__init__()
        self.detection_data = detection_data
        self.crop_bgr = crop_bgr
        self.on_crop_clicked = on_crop_clicked
        self.on_remove_clicked = on_remove_clicked

        layout = QHBoxLayout(self)

        self.crop_label = QLabel()
        self.crop_label.setFixedSize(110, 110)
        self.crop_label.setAlignment(Qt.AlignCenter)
        self.crop_label.setStyleSheet("background-color: white; border: 1px solid #cccccc;")

        pixmap = self._crop_to_pixmap(crop_bgr)
        if pixmap is not None:
            self.crop_label.setPixmap(
                pixmap.scaled(
                    self.crop_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        else:
            self.crop_label.setText("Invalid\ncrop")

        self.crop_label.mousePressEvent = self._handle_crop_click
        layout.addWidget(self.crop_label)

        info_layout = QVBoxLayout()

        top_row = QHBoxLayout()

        self.remove_button = QPushButton("✕")
        self.remove_button.setFixedSize(36, 36)

        self.remove_button.setToolTip("Remove this crop from preview")
        self.remove_button.clicked.connect(self._handle_remove_click)

        top_row.addStretch()
        top_row.addWidget(self.remove_button)

        bbox_text = f"BBox: {self.detection_data['bbox']}"
        detected_text = (
            f"YOLO: {self.detection_data.get('detected_label', 'unknown')} "
            f"({self.detection_data.get('confidence', 0.0):.2f})"
        )

        cls_label = self.detection_data.get("editable_label", "unknown")
        cls_score = self.detection_data.get("classification_score", None)
        if cls_score is not None:
            predicted_text = f"Classifier: {cls_label} ({cls_score:.2f})"
        else:
            predicted_text = f"Classifier: {cls_label}"

        self.detected_label = QLabel(detected_text)
        self.predicted_label = QLabel(predicted_text)
        self.bbox_label = QLabel(bbox_text)

        self.species_combo = QComboBox()
        self.species_combo.setEditable(True)
        self.species_combo.addItems(species_options)

        current_species = cls_label if cls_label else "unknown"
        combo_index = self.species_combo.findText(current_species)
        if combo_index >= 0:
            self.species_combo.setCurrentIndex(combo_index)
        else:
            self.species_combo.addItem(current_species)
            self.species_combo.setCurrentText(current_species)

        info_layout.addLayout(top_row)
        info_layout.addWidget(self.detected_label)
        info_layout.addWidget(self.predicted_label)
        info_layout.addWidget(self.bbox_label)
        info_layout.addWidget(self.species_combo)

        layout.addLayout(info_layout)

    def _crop_to_pixmap(self, crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = crop_rgb.shape
        bytes_per_line = c * w

        image = QImage(
            crop_rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888,
        )
        return QPixmap.fromImage(image)

    def _handle_crop_click(self, event):
        if self.on_crop_clicked is not None:
            self.on_crop_clicked(self.crop_bgr)

    def _handle_remove_click(self):
        if self.on_remove_clicked is not None:
            self.on_remove_clicked(self)

    def get_selected_species(self):
        return sanitize_name(self.species_combo.currentText())

    def _crop_to_pixmap(self, crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = crop_rgb.shape
        bytes_per_line = c * w

        image = QImage(
            crop_rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888,
        )
        return QPixmap.fromImage(image)

    def _handle_crop_click(self, event):
        if self.on_crop_clicked is not None:
            self.on_crop_clicked(self.crop_bgr)

    def get_selected_species(self):
        return sanitize_name(self.species_combo.currentText())

class VideoWindow(QMainWindow):
    """Main Qt window displaying video frames and current preview sidebar."""

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

        # Current preview widgets/data
        self.preview_item_widgets = []
        self.current_preview_count = 0

        # Dataset paths
        self.dataset_root = (Path(__file__).resolve().parent / "../dataset").resolve()
        self.categories_path = self.dataset_root / "species_categories.json"

        # FPS tracking
        self.frame_count = 0
        self.last_fps_timestamp = time.time()
        self.current_fps = 0.0

        # Persisted categories
        self.species_options = []
        
        self.selected_crop_bgr = None
        self.image_preview_window = None

        self.setWindowTitle("Foram detection")
        self.resize(1380, 780)

        self._build_ui()
        self._ensure_dataset_dirs()
        self._load_species_options()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self._init_source()

    def _build_ui(self):
        """Create the UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)

        splitter = QSplitter(Qt.Horizontal)

        # Left panel: video + controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.video_label = QLabel("Initializing video...")
        self.video_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.video_label, stretch=1)

        controls_layout = QHBoxLayout()

        self.play_pause_button = QPushButton()
        self.play_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.play_pause_button.clicked.connect(self.toggle_pause)

        self.save_button = QPushButton("Save selection")
        self.save_button.clicked.connect(self.save_current_selection)

        self.refresh_preview_button = QPushButton("Refresh preview")
        self.refresh_preview_button.clicked.connect(self.rebuild_preview_sidebar)

        controls_layout.addWidget(self.play_pause_button)
        controls_layout.addWidget(self.save_button)
        controls_layout.addWidget(self.refresh_preview_button)

        self.status_label = QLabel("Status: Idle")
        controls_layout.addWidget(self.status_label)

        controls_layout.addStretch()

        self.fps_label = QLabel("FPS: 0.0")
        controls_layout.addWidget(self.fps_label)

        left_layout.addLayout(controls_layout)

        # Right panel: station + preview sidebar
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        station_row = QHBoxLayout()
        station_label = QLabel("Station:")
        self.station_input = QLineEdit()
        self.station_input.setPlaceholderText("e.g. StB_C3_PLN_MTB24_0")
        station_row.addWidget(station_label)
        station_row.addWidget(self.station_input)
        right_layout.addLayout(station_row)

        self.preview_title_label = QLabel("Current preview (0)")
        right_layout.addWidget(self.preview_title_label)
        
        self.selected_crop_preview = QLabel("Click a crop to preview it")
        self.selected_crop_preview.setAlignment(Qt.AlignCenter)
        self.selected_crop_preview.setMinimumHeight(220)
        self.selected_crop_preview.setStyleSheet(
            "background-color: white; border: 1px solid #cccccc;"
        )
        right_layout.addWidget(self.selected_crop_preview)
        
        self.open_preview_window_button = QPushButton("Open in window")
        self.open_preview_window_button.clicked.connect(self.open_selected_crop_in_window)
        right_layout.addWidget(self.open_preview_window_button)
        

        self.preview_scroll_area = QScrollArea()
        self.preview_scroll_area.setWidgetResizable(True)
        right_layout.addWidget(self.preview_scroll_area, stretch=1)

        self.preview_scroll_container = QWidget()
        self.preview_scroll_layout = QVBoxLayout(self.preview_scroll_container)
        self.preview_scroll_layout.setAlignment(Qt.AlignTop)
        self.preview_scroll_area.setWidget(self.preview_scroll_container)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([940, 420])

        root_layout.addWidget(splitter)

    def _ensure_dataset_dirs(self):
        """Create dataset root and categories file if needed."""
        self.dataset_root.mkdir(parents=True, exist_ok=True)

        if not self.categories_path.exists():
            self._write_categories(["unknown"])

    def _read_categories(self):
        if not self.categories_path.exists():
            return ["unknown"]

        with open(self.categories_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_categories(self, categories):
        categories = sorted(set(categories))
        with open(self.categories_path, "w", encoding="utf-8") as f:
            json.dump(categories, f, indent=2, ensure_ascii=False)

    def _load_species_options(self):
        """Load categories from JSON + existing dataset folders."""
        categories = set(self._read_categories())

        if self.dataset_root.exists():
            for item in self.dataset_root.iterdir():
                if item.is_dir():
                    categories.add(item.name)

        categories.add("unknown")
        self.species_options = sorted(categories)
        self._write_categories(self.species_options)

    def _add_species_option(self, species_name: str):
        """Persist a new category if not already known."""
        species_name = sanitize_name(species_name)
        if not species_name:
            return

        if species_name not in self.species_options:
            self.species_options.append(species_name)
            self.species_options = sorted(set(self.species_options))
            self._write_categories(self.species_options)

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
        """Fetch a frame from the source and update display + preview."""
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

        # If model predicts a new species, persist it as a category
        for det in self.last_detections:
            predicted_species = sanitize_name(det.get("editable_label", "unknown"))
            if predicted_species and predicted_species != "unknown":
                self._add_species_option(predicted_species)

        self.current_pixmap = self.frame_to_pixmap(self.last_display_frame)
        self.refresh_display()
        self.update_fps()
        self.rebuild_preview_sidebar()

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

    def _extract_crop(self, frame, bbox):
        """Extract crop from raw frame using bbox."""
        if frame is None:
            return None

        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        x1 = max(0, min(x1, frame_w - 1))
        y1 = max(0, min(y1, frame_h - 1))
        x2 = max(0, min(x2, frame_w))
        y2 = max(0, min(y2, frame_h))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        return crop

    def rebuild_preview_sidebar(self):
        """Rebuild the right sidebar from current detections only."""
        while self.preview_scroll_layout.count():
            child = self.preview_scroll_layout.takeAt(0)
            widget = child.widget()
            if widget is not None:
                widget.deleteLater()

        self.preview_item_widgets = []

        if self.last_raw_frame is None or not self.last_detections:
            self.current_preview_count = 0
            self.preview_title_label.setText("Current preview (0)")
            self.selected_crop_preview.setText("Click a crop to preview it")
            self.selected_crop_preview.setPixmap(QPixmap())
            self.selected_crop_bgr = None
            return

        count = 0
        for det in self.last_detections:
            crop = self._extract_crop(self.last_raw_frame, det["bbox"])
            if crop is None:
                continue

            item_widget = PreviewDetectionItemWidget(
                crop,
                det,
                self.species_options,
                on_crop_clicked=self.show_selected_crop_preview,
                on_remove_clicked=self.remove_preview_item,
            )
            self.preview_scroll_layout.addWidget(item_widget)
            self.preview_item_widgets.append(item_widget)
            count += 1

        self.current_preview_count = count
        self.preview_title_label.setText(f"Current preview ({count})")

        # Auto-display the first crop in the large preview
        if self.preview_item_widgets:
            self.show_selected_crop_preview(self.preview_item_widgets[0].crop_bgr)
        else:
            self.selected_crop_preview.setText("Click a crop to preview it")
            self.selected_crop_preview.setPixmap(QPixmap())

    def _next_available_filename(self, species_name: str, station_name: str):
        """
        Return a filename like:
            Species-Station-ID.png
        where ID increments if needed.
        """
        species_name = sanitize_name(species_name)
        station_name = sanitize_name(station_name)

        species_dir = self.dataset_root / species_name
        species_dir.mkdir(parents=True, exist_ok=True)

        next_id = 1
        while True:
            filename = f"{species_name}-{station_name}-{next_id}.png"
            path = species_dir / filename
            if not path.exists():
                return path
            next_id += 1

    def save_current_selection(self):
        """
        Save only the current preview selection.
        Nothing is autosaved.
        After saving, the sidebar preview is cleared.
        """
        station_name = sanitize_name(self.station_input.text())
        if not station_name:
            QMessageBox.warning(self, "Save selection", "Please enter a station name first.")
            return

        if not self.preview_item_widgets:
            QMessageBox.information(self, "Save selection", "No preview items to save.")
            return

        saved_count = 0

        for item_widget in self.preview_item_widgets:
            species_name = item_widget.get_selected_species()
            crop = item_widget.crop_bgr

            if not species_name:
                continue
            if crop is None or crop.size == 0:
                continue

            self._add_species_option(species_name)

            output_path = self._next_available_filename(species_name, station_name)
            cv2.imwrite(str(output_path), crop)
            saved_count += 1

        QMessageBox.information(
            self,
            "Save selection",
            f"Saved {saved_count} crop(s) for station {station_name}.",
        )

        # Clear current preview after save
        self.last_detections = []
        self.preview_item_widgets = []
        self.rebuild_preview_sidebar()

        # Clear the large preview too
        self.selected_crop_preview.setText("Click a crop to preview it")
        self.selected_crop_preview.setPixmap(QPixmap())
        self.selected_crop_bgr = None
        
        
    def show_selected_crop_preview(self, crop_bgr):
        """Display a clicked crop in the large preview area."""
        self.selected_crop_bgr = crop_bgr

        if crop_bgr is None or crop_bgr.size == 0:
            self.selected_crop_preview.setText("Invalid crop")
            self.selected_crop_preview.setPixmap(QPixmap())
            return

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = crop_rgb.shape
        bytes_per_line = c * w

        image = QImage(
            crop_rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888,
        )

        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.selected_crop_preview.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.selected_crop_preview.setPixmap(scaled)
        
    def open_selected_crop_in_window(self):
        """Open the currently selected crop in a separate large window."""
        if self.selected_crop_bgr is None or self.selected_crop_bgr.size == 0:
            QMessageBox.information(self, "Open preview", "No crop selected.")
            return

        if self.image_preview_window is None:
            self.image_preview_window = ImagePreviewWindow()

        self.image_preview_window.set_crop(self.selected_crop_bgr)
        self.image_preview_window.show()
        self.image_preview_window.raise_()
        self.image_preview_window.activateWindow()
        
    def remove_preview_item(self, item_widget):
        """Remove one crop from the current preview."""
        if item_widget in self.preview_item_widgets:
            self.preview_item_widgets.remove(item_widget)
            item_widget.setParent(None)
            item_widget.deleteLater()

        self.current_preview_count = len(self.preview_item_widgets)
        self.preview_title_label.setText(f"Current preview ({self.current_preview_count})")

        if self.preview_item_widgets:
            self.show_selected_crop_preview(self.preview_item_widgets[0].crop_bgr)
        else:
            self.selected_crop_preview.setText("Click a crop to preview it")
            self.selected_crop_preview.setPixmap(QPixmap())
            self.selected_crop_bgr = None

    def resizeEvent(self, event):
        """Handle window resize events."""
        super().resizeEvent(event)
        self.refresh_display()

    def closeEvent(self, event):
        """Release resources when closing the window."""
        self.timer.stop()
        self.source.release()
        super().closeEvent(event)