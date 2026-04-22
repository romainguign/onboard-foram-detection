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
            # background-color: #1976d2;
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
    
    

    yolo_model = YOLO("../Models/sharp_living/weights/best.pt")
    detector = YoloDetector(yolo_model)

    
    
    # SOURCE OPTIONS:
    
    # A video file source 
    base_source = FileVideoSource("../data/testvideo.mp4", loop=True)
    
    # A folder of images source
    # base_source = ImageFolderSource("../data/images", interval_seconds=3.0)

    # A webcam source
    # base_source = OpenCVVideoSource(0)

    
    source = AnnotatedVideoSource(base_source, detector)
    
    

    window = VideoWindow(source)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()