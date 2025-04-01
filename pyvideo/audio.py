# audio.py

import os
from pathlib import Path
from typing import Self, Generator, Iterable
from dataclasses import dataclass
from abc import ABCMeta
from itertools import cycle

import numpy as np
import cv2
from moviepy import AudioFileClip, AudioArrayClip

from pyvideo.base import TimedFrames, TimedFramesArray, TimedFramesList, action


__all__ = [
    "BaseAudio",
    "AudioList",
    "AudioArray",
    "volume"
]


@dataclass
class BaseAudio(TimedFrames, metaclass=ABCMeta):
    """A class to represent data of audio file or audio of video file."""

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

        self.fps = audio.fps

        for chunk in audio.iter_chunks(chunksize=chunk_size):
            for frame in chunk:
                yield frame

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

        self.set_frames(list(self.read_frames(path=path, chunk_size=chunk_size)))

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
        cap.release()

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

    def audio_array(self) -> "AudioArray":
        return AudioArray(fps=self.fps, frames=self.array())

    def audio_list(self) -> "AudioList":
        return AudioList(fps=self.fps, frames=list(self.data_copy()))


@dataclass
class AudioList(BaseAudio, TimedFramesList):
    """A class to represent data of audio file or audio of video file."""


@dataclass
class AudioArray(BaseAudio, TimedFramesArray):
    """A class to represent data of audio file or audio of video file."""


def volume(audio: BaseAudio, factor: float | np.ndarray, deep: bool = False) -> Iterable[np.ndarray]:
    if isinstance(factor, (int, float)):
        f = factor
        factor = (f for _ in range(len(audio)))

    if deep:
        factor = cycle(factor)

    factor = iter(factor)

    yield from action(
        audio,
        change_data=lambda frame: frame * next(factor),
        deep=deep
    )
