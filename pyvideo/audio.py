# audio.py

import os
from pathlib import Path
from typing import Self, Generator, Iterable, Callable

import numpy as np
import cv2
from moviepy import AudioFileClip, AudioArrayClip

from pyvideo.utils import ManagedModel


__all__ = [
    "Audio"
]


class Audio(ManagedModel[np.ndarray]):
    """A class to represent data of audio file or audio of video file."""

    def __init__(
        self,
        fps: float,
        source: str | Path = None,
        destination: str | Path = None,
        frames: list[np.ndarray] = None,
        resolution: int = 12
    ) -> None:
        """
        Defines the attributes of a video.

        :param fps: The frames per second rate.
        :param source: The source file path.
        :param destination: The destination file path.
        :param frames: The list of frames.
        :param resolution: The accuracy of floating point numbers.
        """
        super().__init__()

        self.fps = fps
        self.resolution = resolution

        self.source = source
        self.destination = destination

        self.frames = [] if frames is None else frames

        self.reset_tasks()

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

    def data(self) -> Iterable[np.ndarray]:
        return iter(self.frames)

    def all(self) -> Self:
        return super().all()

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

    def select(
        self,
        start: int = None,
        end: int = None,
        step: int = None,
        selector: Callable[[int, np.ndarray], bool] = None
    ) -> Self:
        """
        Cuts the video.

        :param start: The starting index for the frames.
        :param end: The ending index for the frames.
        :param step: The step for the frames.
        :param selector: The function to select frames.
        :return: The modified video object.
        """

        start = start or 0
        end = end or self.length
        step = step or 1

        if not self._manager.is_empty:
            raise ValueError(
                "This error prevents the discarding of existing tasks "
                "pending for the object. Selecting overrides existing tasks. "
                "If this is intended, use the empty method to empty the "
                "manager in advance."
            )

        self.reset_tasks(
            (
                frame for i, frame in enumerate(self.frames[start:end:step])
                if selector is None or selector(i, frame)
            )
        )

        return self

    def volume(self, factor: float) -> Self:
        """
        Changes the volume of the audio.

        :param factor: The change value.

        :return: The changes audio object.
        """

        frames = []

        def end():
            self.frames[:] = frames

        self._manager.load(
            repeat=lambda frame: frame * factor,
            end=end, collector=frames
        )

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
        path: str | Path = None,
        chunk_size: int | None = 50000
    ) -> Generator[np.ndarray, None, None]:
        """
        Loads the audio data from the file.

        :param path: The path to the source file.
        :param chunk_size: The chunk size of each read.

        :return: The loaded file data.
        """

        chunk_size = 50000 if chunk_size is None else chunk_size

        path = path or self.source

        if path is None:
            raise ValueError("No path specified.")

        path = str(path)

        audio = AudioFileClip(path)

        frames = []

        for chunk in audio.iter_chunks(chunksize=chunk_size):
            for frame in chunk:
                yield frame

                frames.append(frame)

        self.fps = audio.fps

        audio.close()

        self.source = path

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

        self.frames.extend(self.read_frames(path=path, chunk_size=chunk_size))

    @classmethod
    def load(
        cls,
        path: str | Path,
        destination: str | Path = None,
        frames: Iterable[np.ndarray] = None,
        chunk_size: int | None = 50000,
        start: int = None,
        end: int = None,
        step: int = None
    ) -> Self:
        """
        Loads the data from the file.

        :param path: The path to the source file.
        :param destination: The destination to set for the video object.
        :param frames: The frames to insert to the video data object.
        :param chunk_size: The chunk size of each read.
        :param start: The starting index for the frames.
        :param end: The ending index for the frames.
        :param step: The step for the frames.

        :return: The loaded file data.
        """

        path = str(path)

        cap = cv2.VideoCapture(path)

        fps = float(cap.get(cv2.CAP_PROP_FPS))

        frames = [] if frames is None else frames

        audio = cls(
            frames=frames, fps=fps,
            source=path, destination=destination
        )

        audio.load_frames(path=path, chunk_size=chunk_size)
        audio.cut(start=start, end=end, step=step)

        return audio

    def save(self, path: str | Path = None) -> None:
        """
        Saves the video and audio into the file.

        :param path: The saving path
        """

        path = path or self.destination

        if path is None:
            raise ValueError("No path specified.")

        path = str(path)

        if location := Path(path).parent:
            os.makedirs(location, exist_ok=True)

        audio = self.moviepy()
        audio.write_audiofile(path, logger=None)
        audio.close()

        self.destination = path

    def moviepy(self) -> AudioArrayClip:
        return AudioArrayClip(np.array(self.frames), fps=self.fps)

    def copy(self) -> Self:
        """Creates a copy of the data."""

        return Audio(
            frames=[frame.copy() for frame in self.frames],
            fps=self.fps,
            source=self.source,
            destination=self.destination
        )
