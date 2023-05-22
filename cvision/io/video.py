from typing import Union, List, Tuple
from pathlib import Path
import os
import cv2
import numpy as np
from .video_reader import VideoReader
from .video_meta import VideoMetaData


class Video:
    """
    A class for reading and extracting frames from a video file.

    Methods:
        __len__(): Returns the total number of frames in the video.
        get_metadata(): Returns the metadata of the video.
        save_frames(output_dir, img_format): Extracts and saves video frames to disk in the specified image format.
        extract_frames(output_dir, interval): Extracts video frames in the specified interval and return as list.
    """
    def __init__(self,
                 path: Union[str, Path]
                 ) -> None:
        """
        Initializes a new instance of the `Video` class.

        Args:
            path (Union[str, Path]): A path to the video file.
        """
        if isinstance(path, Path):
            path = path.as_posix()

        if not os.path.exists(path):
            raise RuntimeError(f"File not found: {path}")

        self.path = path
        self.metadata = VideoMetaData.from_path(path=self.path)

    def __len__(self) -> int:
        """
        Returns the total number of frames in the video.

        Returns:
            int: The total number of frames in the video.
        """
        return self.metadata.frame_count

    def get_metadata(self):
        """
        Returns the metadata of the video.

        Returns:
            VideoMetaData: The metadata of the video.
        """
        return self.metadata

    def save_frames(self, output_dir: Union[str, Path], img_format: str = "png"):
        """
        Extracts and saves video frames to disk in the specified image format.

        Args:
            output_dir (Union[str, Path]): The directory to save the frames to.
            img_format (str, optional): The image format to save the frames.

        Returns:
            None
        """
        os.makedirs(output_dir, exist_ok=True)
        with VideoReader(self.path) as reader:
            for index, frame in enumerate(reader):
                filename = os.path.join(output_dir, f"{index}.{img_format}")
                cv2.imwrite(filename, frame)

    def _get_steps(self, stepsize) -> int:
        """
        Returns the number of steps to take when iterating over the video frames.
        
        Args:
            stepsize (int, optional): The step size to take when iterating over the video frames.
            
        Returns:
            int: The number of steps to take when iterating over the video frames.
        """
        if stepsize is None:
            return 1
        return int(stepsize * self.metadata.fps)

    def extract_frames(self, stepsize=None) -> List[np.ndarray]:
        """
        Extracts frames from the video in the instance's `path`, and returns them as a list of NumPy arrays.
        
        Args:
            stepsize (int, optional): The step size to take when iterating over the video frames.

        Returns:
            List[np.ndarray]: A list of frames as NumPy arrays.
        """
        frames = []
        steps = self._get_steps(stepsize)
        with VideoReader(self.path) as reader:
            for frame_id in range(0, self.metadata.frame_count, steps):
                reader.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = reader.cap.read()
                if not ret:
                    break
                frames.append(frame)
        return frames
