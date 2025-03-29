# audio.py

import os
from pathlib import Path
from typing import Self, Generator, Iterable
from dataclasses import dataclass

import numpy as np
import cv2
from moviepy import AudioFileClip, AudioArrayClip

from pyvideo2.action import TimedFrames


__all__ = [
    "Audio"
]


@dataclass
class Audio(TimedFrames):
    """A class to represent data of audio file or audio of video file."""

    def volume(self, factor: float) -> Self:
        """
        Changes the volume of the audio.

        :param factor: The change value.

        :return: The changes audio object.
        """

        self.frames[:] = np.array(self.frames) * factor

        return self

    def read_frames(
        self,
        path: str | Path,
        chunk_size: int | None = 50000
    ) -> Generator[np.ndarray, None, None]:
        """
        Loads the audio data from the file.

        :param path: The path to the source file.
        :param chunk_size: The chunk size of each read.

        :return: The loaded file data.
        """

        chunk_size = 50000 if chunk_size is None else chunk_size

        audio = AudioFileClip(str(path))

        for chunk in audio.iter_chunks(chunksize=chunk_size):
            for frame in chunk:
                yield frame

        self.fps = audio.fps

        audio.close()

    def load_frames(
        self,
        path: str | Path,
        chunk_size: int | None = 50000
    ) -> None:
        """
        Loads the video data from the file.

        :param path: The path to the source file.
        :param chunk_size: The chunk size of each read.
        """

        self.frames.extend(
            self.read_frames(path=path, chunk_size=chunk_size)
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        frames: Iterable[np.ndarray] = None,
        chunk_size: int | None = 50000
    ) -> Self:
        """
        Loads the data from the file.

        :param path: The path to the source file.
        :param frames: The frames to insert to the video data object.
        :param chunk_size: The chunk size of each read.

        :return: The loaded file data.
        """

        cap = cv2.VideoCapture(str(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS))

        audio = cls(frames=[] if frames is None else frames, fps=fps)
        audio.load_frames(path=path, chunk_size=chunk_size)

        return audio

    def save(self, path: str | Path) -> None:
        """
        Saves the video and audio into the file.

        :param path: The saving path
        """

        path = str(path)

        if location := Path(path).parent:
            os.makedirs(location, exist_ok=True)

        audio = self.moviepy()
        audio.write_audiofile(path, logger=None)
        audio.close()

    def moviepy(self) -> AudioArrayClip:
        return AudioArrayClip(self.array(), fps=self.fps)

    def copy(self) -> Self:
        """Creates a copy of the data."""

        return Audio(
            frames=[frame.copy() for frame in self.frames],
            fps=self.fps
        )
