# audio.py

import os
from pathlib import Path
from typing import Self, Generator, Iterable
from dataclasses import dataclass, field

import numpy as np
import cv2
from moviepy import AudioFileClip, AudioArrayClip


__all__ = [
    "Audio"
]


@dataclass
class Audio:
    """A class to represent data of audio file or audio of video file."""

    fps: float
    frames: list[np.ndarray] = field(default_factory=list)
    resolution: int = 12

    @property
    def length(self) -> int:
        """
        Returns the amount of frames in the video.

        :return: The int amount of frames.
        """

        return len(self.frames)

    @property
    def duration(self) -> float:
        """
        Returns the amount of time in the video.

        :return: The int amount of time.
        """

        return round(self.length / self.fps, self.resolution)

    @property
    def span(self) -> float:
        """
        Returns the duration divided by the length.

        :return: The int span of the data.
        """

        return 1 / self.fps

    def time_frame(self) -> list[float]:
        """
        Returns a list of the time points.

        :return: The list of time points.
        """

        return [self.time(i) for i in range(self.length)]

    def time(self, index: int) -> float:
        return round(index * self.span, self.resolution)

    def index(self, time: float) -> int:
        return np.round(time / self.span).astype(int)

    def cut(
            self,
            start: int = None,
            end: int = None,
            step: int = None
    ) -> Self:
        """
        Cuts the video.

        :param start: The starting index for the frames.
        :param end: The ending index for the frames.
        :param step: The step for the frames.
        :return: The modified video object.
        """

        start = start or 0
        end = end or self.length
        step = step or 1

        self.frames[:] = self.frames[start:end:step]

        return self

    def volume(self, factor: float) -> Self:
        """
        Changes the volume of the audio.

        :param factor: The change value.

        :return: The changes audio object.
        """

        self.frames[:] = np.array(self.frames) * factor

        return self

    def speed(self, factor: float) -> Self:
        """
        Changes the speed of the playing.

        :param factor: The speed factor.

        :return: The changes audio object.
        """

        self.fps *= factor

        return self

    def read_frames(
        self,
        path: str | Path,
        chunk_size: int | None = 50000,
        start: int = None,
        end: int = None,
        step: int = None
    ) -> Generator[np.ndarray, None, None]:
        """
        Loads the audio data from the file.

        :param path: The path to the source file.
        :param chunk_size: The chunk size of each read.
        :param start: The starting index for the frames.
        :param end: The ending index for the frames.
        :param step: The step for the frames.

        :return: The loaded file data.
        """

        chunk_size = 50000 if chunk_size is None else chunk_size

        audio = AudioFileClip(str(path))

        frames = []

        i = 0
        iterations = iter(range(start, end, step))

        for chunk in audio.iter_chunks(chunksize=chunk_size):
            for frame in chunk:
                if i == next(iterations):
                    yield frame
                    frames.append(frame)

                i += 1

        self.fps = audio.fps

        audio.close()

    def load_frames(
        self,
        path: str | Path,
        chunk_size: int | None = 50000,
        start: int = None,
        end: int = None,
        step: int = None
    ) -> None:
        """
        Loads the video data from the file.

        :param path: The path to the source file.
        :param chunk_size: The chunk size of each read.
        :param start: The starting index for the frames.
        :param end: The ending index for the frames.
        :param step: The step for the frames.
        """

        self.frames.extend(
            self.read_frames(
                path=path, chunk_size=chunk_size,
                start=start, end=end, step=step
            )
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        frames: Iterable[np.ndarray] = None,
        chunk_size: int | None = 50000,
        start: int = None,
        end: int = None,
        step: int = None
    ) -> Self:
        """
        Loads the data from the file.

        :param path: The path to the source file.
        :param frames: The frames to insert to the video data object.
        :param chunk_size: The chunk size of each read.
        :param start: The starting index for the frames.
        :param end: The ending index for the frames.
        :param step: The step for the frames.

        :return: The loaded file data.
        """

        cap = cv2.VideoCapture(str(path))

        fps = float(cap.get(cv2.CAP_PROP_FPS))

        frames = [] if frames is None else frames

        audio = cls(frames=frames, fps=fps)

        audio.load_frames(
            path=path, chunk_size=chunk_size,
            start=start, end=end, step=step
        )

        return audio

    def save(self, path: str | Path) -> None:
        """
        Saves the video and audio into the file.

        :param path: The saving path
        """

        if path is None:
            raise ValueError("No path specified.")

        path = str(path)

        if location := Path(path).parent:
            os.makedirs(location, exist_ok=True)

        audio = self.moviepy()
        audio.write_audiofile(path, logger=None)
        audio.close()

    def array(self) -> np.ndarray:
        return np.array(self.frames)

    def moviepy(self) -> AudioArrayClip:
        return AudioArrayClip(self.array(), fps=self.fps)

    def copy(self) -> Self:
        """Creates a copy of the data."""

        return Audio(
            frames=[frame.copy() for frame in self.frames],
            fps=self.fps
        )
