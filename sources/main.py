import sys
from PySide6.QtWidgets import QApplication
from ultralytics import YOLO

from detector import YoloDetector
from video_sources import FileVideoSource, AnnotatedVideoSource, ImageFolderSource, OpenCVVideoSource
from windowUI import VideoWindow


def main():
    app = QApplication(sys.argv)

    app.setStyleSheet("""
        QMainWindow {
            background-color: #f7fff9;
        }

        QLabel {
            color: black;
            font-size: 14px;
        }

        QPushButton {
            color: black;
            border-radius: 6px;
            padding: 6px 12px;
        }

        QPushButton:hover {
            background-color: #dceffd;
        }

        QPushButton:pressed {
            background-color: #c6e2fb;
        }

        QScrollArea {
            background-color: white;
        }

        QComboBox {
            padding: 4px;
        }
    """)

    yolo_model = YOLO("../Models/sharp_living/weights/best.pt")
    detector = YoloDetector(yolo_model)

    # Video file source
    # base_source = FileVideoSource("../data/testvideo.mp4", loop=True)

    # Folder of images source
    base_source = ImageFolderSource("../data/images", interval_seconds=3.0)

    # Webcam source
    # base_source = OpenCVVideoSource(0)

    source = AnnotatedVideoSource(base_source, detector)

    window = VideoWindow(source)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()