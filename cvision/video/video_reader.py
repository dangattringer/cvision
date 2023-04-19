from __future__ import annotations
from typing import Any, Dict, Iterator, List, Union
import traceback
import cv2
import numpy as np
from .video_meta import VideoMetaData


class VideoReader:
    """
    A class for reading and extracting frames from video files.

    Attributes:
        path (str): The path to the video file.
        interval (int): The number of seconds between each frame. Defaults to 1.
        start_time (int): The start time of the video in seconds. Defaults to 0.
        metadata (VideoMetaData): The metadata of the video file.

    Methods:
        __enter__(): Returns the instance of the VideoReader class.
        __exit__(exc_type, exc_val, exc_tb): Releases the video capture and prints any exception traceback if exists.
        __len__(): Returns the total number of frames in the video.
        get_steps(): Returns the number of frames to skip in order to obtain the desired frame interval.
        extract_frames(): Extracts and returns all frames of the video.
        generator(): Generator that iterates over the frames of the video file. Each iteration returns a frame as a 
            NumPy array.
    """
    def __init__(self,
                 path: str,
                 interval: int = 1,
                 start_time: int = 0) -> None:
        """
        Initializes a new instance of the `VideoReader` class.

        Args:
            path (str): The path to the video file.
            interval (int, optional): The desired frame interval. Only every `interval`-th frame will be returned
                by the `generator` and `extract_frames` methods. Defaults to 1.
            start_time (int, optional): The time in seconds to start reading the video from. Defaults to 0.
        """
        self.path = str(path)
        self.interval = interval
        self.start_time = start_time
        self.metadata = VideoMetaData.from_path(path=self.path)

    def __enter__(self) -> VideoReader:
        """
        Context manager enter method. Returns the instance of the VideoReader class.
        """
        self.cap = cv2.VideoCapture(self.path)
        if self.start_time > 0:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, self.start_time * 1000)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Context manager exit method. Releases the video capture and prints any exception traceback if exists.

        Args:
            exc_type (Type): The type of the exception raised.
            exc_val (Exception): The exception instance raised.
            exc_tb (Traceback): The traceback object representing the call stack at the point where the exception 
                originally occurred.
        """
        if self.cap is not None:
            self.cap.release()
        if exc_type is not None:
            print(f"Error occurred: {exc_val}")
            traceback.print_exception(exc_type, exc_val, exc_tb)

    def __len__(self) -> int:
        """
        Returns the total number of frames in the video.

        Returns:
            int: The total number of frames in the video.
        """
        return self.metadata.frame_count

    def get_steps(self) -> int:
        """
        Returns the number of frames to skip in order to obtain the desired frame interval.

        Returns:
            int: The number of frames to skip.
        """
        return int(self.interval * self.metadata.fps)

    def extract_frames(self) -> List[np.ndarray]:
        """
        Extracts and returns all frames of the video.

        If the `interval` attribute of the instance is set to a positive integer,
        only every `interval`-th frame is returned. Otherwise, all frames are returned.

        Returns:
            A list of numpy arrays, each representing a single frame of the video.
        """
        frames = []
        steps = self.get_steps()
        for frame_id in range(0, self.metadata.frame_count, steps):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame)
        self.cap.release()
        return frames

    def generator(self) -> Iterator[np.ndarray]:
        """
        Generator that iterates over the frames of the video file.
        Each iteration returns a frame as a NumPy array.

        If the `interval` attribute of the instance is set to a positive integer,
        only every `interval`-th frame is returned. Otherwise, all frames are returned.

        Returns:
            Iterator[np.ndarray]: A generator of NumPy arrays representing the video frames.
        """
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

            if self.interval is None:
                continue
            else:
                steps = self.get_steps()
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cap.get(
                    cv2.CAP_PROP_POS_FRAMES) + steps)
