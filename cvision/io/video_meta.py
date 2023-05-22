import cv2
from typing import NamedTuple


class VideoMetaData(NamedTuple):
    fps: float
    frame_count: int
    duration: float
    format: str
    fourcc: int
    width: int
    height: int

    @classmethod
    def from_path(cls, path: str):
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        format = cap.get(cv2.CAP_PROP_FORMAT)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return VideoMetaData(fps,
                            frame_count,
                            duration,
                            format,
                            fourcc,
                            width,
                            height)
