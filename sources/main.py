import sys
from PySide6.QtWidgets import QApplication

from video_sources import FileVideoSource, OpenCVVideoSource
from windowUI import VideoWindow


def main():
    app = QApplication(sys.argv)

    # Development with a file
    source = FileVideoSource("../data/testvideo.mp4", loop=True)

    # Webcam
    # source = OpenCVVideoSource(0)

    window = VideoWindow(source)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()