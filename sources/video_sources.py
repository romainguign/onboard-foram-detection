from abc import ABC, abstractmethod
from pathlib import Path

import cv2


class VideoSource(ABC):
    """Abstract base class for any video source."""

    @abstractmethod
    def open(self) -> bool:
        pass

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def release(self) -> None:
        pass

    def get_fps(self) -> float:
        return 30.0


class FileVideoSource(VideoSource):
    """Video source based on a file."""

    def __init__(self, path: str, loop: bool = True):
        self.path = path
        self.loop = loop
        self.cap = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.path)
        return self.cap.isOpened()

    def read(self):
        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if ret:
            return True, frame

        if self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return self.cap.read()

        return False, None

    def release(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def get_fps(self) -> float:
        if self.cap is None:
            return 30.0

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0 or fps > 240:
            return 30.0
        return fps


class OpenCVVideoSource(VideoSource):
    """Video source using OpenCV (e.g. webcam)."""

    def __init__(self, device=0):
        self.device = device
        self.cap = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.device)
        return self.cap.isOpened()

    def read(self):
        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def get_fps(self) -> float:
        if self.cap is None:
            return 30.0

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0 or fps > 240:
            return 30.0
        return fps


class ImageFolderSource(VideoSource):
    """
    Image slideshow source.
    Loads one image at a time from a folder, optionally runs a detector,
    and returns the annotated image as a frame.
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, folder_path: str, detector=None, loop: bool = True, interval_seconds: float = 3.0):
        self.folder_path = Path(folder_path)
        self.detector = detector
        self.loop = loop
        self.interval_seconds = interval_seconds

        self.image_paths = []
        self.index = 0

    def open(self) -> bool:
        if not self.folder_path.exists() or not self.folder_path.is_dir():
            return False

        self.image_paths = sorted(
            [
                path for path in self.folder_path.iterdir()
                if path.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ]
        )

        self.index = 0
        return len(self.image_paths) > 0

    def read(self):
        if not self.image_paths:
            return False, None

        if self.index >= len(self.image_paths):
            if self.loop:
                self.index = 0
            else:
                return False, None

        image_path = self.image_paths[self.index]
        frame = cv2.imread(str(image_path))

        self.index += 1

        if frame is None:
            return False, None

        if self.detector is not None:
            frame = self.detector.annotate(frame)

        return True, frame

    def release(self) -> None:
        pass

    def get_fps(self) -> float:
        if self.interval_seconds <= 0:
            return 1.0
        return 1.0 / self.interval_seconds