import sys
from PySide6.QtWidgets import QApplication

from video_sources import FileVideoSource, OpenCVVideoSource
from windowUI import VideoWindow


def main():
    app = QApplication(sys.argv)

    app.setStyleSheet("""
        QMainWindow {
            background-color: ##fffafa
;
        }

        QLabel {
            color: black;
            font-size: 14px;
        }

        QPushButton {
            background-color: #1976d2;
            color: white;
            border-radius: 6px;
            padding: 6px 12px;
        }

        QPushButton:hover {
            background-color: rgb(25, 118, 210, 0.8);
        }

    """)

    source = FileVideoSource("../data/testvideo.mp4", loop=True)


    window = VideoWindow(source)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()