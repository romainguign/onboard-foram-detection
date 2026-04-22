import time
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
)


class VideoWindow(QMainWindow):
    """Main Qt window displaying video frames and playback controls."""

    def __init__(self, source):
        super().__init__()

        self.source = source
        self.current_pixmap = None
        self.is_paused = False
        self.interval_ms = 33

        # FPS tracking
        self.frame_count = 0
        self.last_fps_timestamp = time.time()
        self.current_fps = 0.0

        self.setWindowTitle("Video Stream")
        self.resize(960, 720)

        self._build_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self._init_source()

    def _build_ui(self):
        """Create the UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        self.video_label = QLabel("Initializing video...")
        self.video_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.video_label, stretch=1)

        controls_layout = QHBoxLayout()

        self.play_pause_button = QPushButton("Pause")
        self.play_pause_button.clicked.connect(self.toggle_pause)
        controls_layout.addWidget(self.play_pause_button)

        self.status_label = QLabel("Status: Idle")
        controls_layout.addWidget(self.status_label)

        controls_layout.addStretch()

        self.fps_label = QLabel("FPS: 0.0")
        controls_layout.addWidget(self.fps_label)

        main_layout.addLayout(controls_layout)

    def _init_source(self):
        """Open the video source and start the update timer."""
        if not self.source.open():
            self.video_label.setText("Failed to open video source")
            self.status_label.setText("Status: Error")
            self.play_pause_button.setEnabled(False)
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
            self.play_pause_button.setText("Pause")
            self.status_label.setText("Status: Playing")
        else:
            self.timer.stop()
            self.is_paused = True
            self.play_pause_button.setText("Play")
            self.status_label.setText("Status: Paused")

    def update_frame(self):
        """Fetch a frame from the source and update the display."""
        ret, frame = self.source.read()

        if not ret or frame is None:
            self.timer.stop()
            self.status_label.setText("Status: End / Error")
            self.video_label.setText("End of stream or read error")
            self.play_pause_button.setEnabled(False)
            return

        self.current_pixmap = self.frame_to_pixmap(frame)
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

    def resizeEvent(self, event):
        """Handle window resize events."""
        super().resizeEvent(event)
        self.refresh_display()

    def closeEvent(self, event):
        """Release resources when closing the window."""
        self.timer.stop()
        self.source.release()
        super().closeEvent(event)