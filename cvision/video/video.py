from typing import Union, List, Tuple
from pathlib import Path
import cv2
import numpy as np
from .video_reader import VideoReader


class Video:
    """
    A class for working with video files.

    Attributes:
        paths (List[Union[str, Path]]): A list of paths to the video files.

    Methods:
        save_frames(output_dir, interval=1, img_format="png"):
            Extracts and saves video frames to disk in the specified image format.

        extract_frames(interval=1):
            Extracts frames from the videos in the instance's `paths` list, at a given interval.
    """

    def __init__(self,
                 paths: Union[str, Path, List[Union[str, Path]]]
                 ) -> None:
        """
        Initializes a `Video` instance with one or more video file paths.

        Args:
            paths (Union[str, Path, List[Union[str, Path]]]): A string or list of strings representing the path(s) to the video file(s).

        Returns:
            None
        """
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.paths = paths

    def save_frames(self, output_dir: Union[str, Path], interval=1, img_format: str = "png"):
        """
        Extracts and saves video frames to disk in the specified image format.

        Args:
            output_dir (Union[str, Path]): The directory to save the extracted frames to.
            interval (int, optional): The frame interval to extract. 
            img_format (str, optional): The image format to save the extracted frames in.

        Returns:
            None
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in self.paths:
            with VideoReader(path, interval=interval) as reader:
                for index, frame in enumerate(reader.generator()):
                    filename = output_dir / f"{index}.{img_format}"
                    cv2.imwrite(filename.as_posix(), frame)

    def extract_frames(self, interval=1) -> List[Tuple[str, List[np.ndarray]]]:
        """
        Extracts frames from the videos in the instance's `paths` list, at a given interval.

        Args:
            interval (int, optional): The interval at which to extract frames.

        Returns:
            List[Tuple[str, List[np.ndarray]]]: A list of tuples, where each tuple contains the path of a video and a list
            of numpy arrays representing the extracted frames.
        """
        result = []
        for path in self.paths:
            with VideoReader(path, interval=interval) as reader:
                frames = reader.extract_frames()
            result.append((path, frames))
        return result
