import sys
from PySide6.QtWidgets import QApplication
from ultralytics import YOLO

from detector import YoloDetector
from video_sources import ImageFolderSource
from windowUI import VideoWindow


def main():
    app = QApplication(sys.argv)

    app.setStyleSheet("""
        QMainWindow {
            background-color: #121212;
        }

        QLabel {
            color: white;
            font-size: 14px;
        }

        QPushButton {
            background-color: #1976d2;
            color: white;
            border-radius: 6px;
            padding: 6px 12px;
        }

        QPushButton:hover {
            background-color: #1e88e5;
        }

        QPushButton:pressed {
            background-color: #1565c0;
        }
    """)

    model = YOLO("../Models/sharp_living/weights/best.pt")
    detector = YoloDetector(model)

    source = ImageFolderSource(
        folder_path="../data/images",
        detector=detector,
        loop=True,
        interval_seconds=3.0,
    )

    window = VideoWindow(source)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()