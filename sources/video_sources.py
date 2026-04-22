from abc import ABC, abstractmethod
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
    """Video source using OpenCV (e.g., webcam)."""

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