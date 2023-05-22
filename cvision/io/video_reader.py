from __future__ import annotations
from typing import Any, Dict, Iterator, List, Union
import traceback
import cv2
import numpy as np


class VideoReader:
    """
    A class for reading and extracting frames from a video file.

    Methods:
        __enter__(): Returns the instance of the VideoReader class.
        __exit__(exc_type, exc_val, exc_tb): Releases the video capture and prints any exception traceback if exists.
        __iter__(): Generator that iterates over the frames of the video file. Each iteration returns a frame as a  
            NumPy array.
    """
    def __init__(self,
                 path: str
                 ) -> None:
        """
        Initializes a new instance of the `VideoReader` class.

        Args:
            path (str): The path to the video file.
        """
        self.path = path

    def __enter__(self) -> VideoReader:
        """
        Context manager enter method. Returns the instance of the VideoReader class.
        """
        self.cap = cv2.VideoCapture(self.path)
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

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Generator that iterates over the frames of the video file. Each iteration returns a frame as a NumPy array.
        """
        return self
    
    def __next__(self) -> np.ndarray:
        """
        Returns the next frame of the video file as a NumPy array.
        """
        ret, frame = self.cap.read()
        
        if ret:
            return frame
