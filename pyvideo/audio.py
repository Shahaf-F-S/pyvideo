# audio.py

import os
from pathlib import Path
from typing import ClassVar, Self, Generator, Iterable

import numpy as np
from tqdm import tqdm
import cv2
from moviepy.editor import AudioFileClip, AudioClip

__all__ = [
    "Audio"
]

class Audio:
    """A class to represent data of audio file or audio of video file."""

    SILENT: ClassVar[bool] = True

    def __init__(
            self,
            fps: float,
            source: str | Path = None,
            destination: str | Path = None,
            silent: bool = True,
            frames: list[np.ndarray] = None,
            audio: AudioClip = None
    ) -> None:
        """
        Defines the attributes of a video.

        :param fps: The frames per second rate.
        :param source: The source file path.
        :param destination: The destination file path.
        :param silent: The value to silent output.
        :param frames: The list of frames.
        :param audio: The base audio object.
        """

        if silent is None:
            silent = self.SILENT

        if frames is None:
            frames = []

        self._fps = fps

        self.source = source
        self.destination = destination

        self.silent = silent

        self.frames = frames

        self._audio: AudioFileClip | None = audio

    @property
    def length(self) -> float:
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

        return round(self.length / self._fps, 12)

    @property
    def fps(self) -> float:
        """
        Returns the frame per second rate of the video.

        :return: The video speed.
        """

        return self._fps

    @fps.setter
    def fps(self, value: float) -> None:
        """
        Returns the frame per second rate of the video.

        :param value: The video speed.
        """

        before = self._fps

        self._fps = value

        if isinstance(self._audio, AudioClip):
            self._audio.fps *= (value / before)
            self._audio.duration = round(self.length / self._fps, 12)

    def time_frame(self) -> list[float]:
        """
        Returns a list of the time points.

        :return: The list of time points.
        """

        return [
            round(i * (self.duration / len(self.frames)), 12)
            for i in range(1, len(self.frames) + 1)
        ]

    def cut(
            self,
            start: int = None,
            end: int = None,
            step: int = None,
            inplace: bool = False
    ) -> Self:
        """
        Cuts the video.

        :param start: The starting index for the frames.
        :param end: The ending index for the frames.
        :param step: The step for the frames.
        :param inplace: The value to set changes in the object.

        :return: The modified video object.
        """

        if inplace:
            audio = self

        else:
            audio = self.copy()

        start = start or 0
        end = end or audio.length
        step = step or 1

        if audio.frames:
            audio.frames[:] = audio.frames[start:end:step]

        audio._audio = audio._audio.subclip(
            t_start=round(start * (self.duration / len(self.frames)), 12),
            t_end=round(end * (self.duration / len(self.frames)), 12)
        )

        audio._update_audio()

        return audio

    def _make_frame(self, t: float | np.ndarray) -> np.ndarray | Iterable[np.ndarray]:
        """
        Returns the frame or frames of audio for the given time.

        :param t: The time of the audio.

        :return: The frame or frames of audio.
        """

        _frames = np.array(self.frames)

        if isinstance(t, np.ndarray):
            array_indexes = (self.fps * t).astype(int)
            in_array = (array_indexes > 0) & (array_indexes < len(_frames))
            result = np.zeros((len(t), 2))
            result[in_array] = _frames[array_indexes[in_array]]

            return result

        else:
            # noinspection PyShadowingNames
            i = int(self.fps * t)

            if i < 0 or i >= len(_frames):
                return 0 * _frames[0]

            else:
                return _frames[i]

    def _update_audio(self) -> None:
        """Updates the audio data of the object."""

        self._audio.reader.make_frame = self._make_frame

    def volume(self, factor: float, inplace: bool = False) -> Self:
        """
        Changes the volume of the audio.

        :param factor: The change value.
        :param inplace: The value to save changes to the object.

        :return: The changes audio object.
        """

        if inplace:
            audio = self

        else:
            audio = self.copy()

        audio.frames[:] = [frame * factor for frame in audio.frames]
        audio._audio = audio._audio.fl(
            lambda gf, t: gf(t) * factor, keep_duration=True
        )

        return audio

    def speed(self, factor: float, inplace: bool = False) -> Self:
        """
        Changes the speed of the playing.

        :param factor: The speed factor.
        :param inplace: The value to save changes to the object.

        :return: The changes audio object.
        """

        if inplace:
            audio = self

        else:
            audio = self.copy()

        audio.fps *= factor

        return audio

    def load_frames_generator(
            self,
            path: str | Path = None,
            silent: bool = None,
            start: int = None,
            end: int = None,
            step: int = None
    ) -> Generator[np.ndarray, None, None]:
        """
        Loads the audio data from the file.

        :param path: The path to the source file.
        :param silent: The value for no output.
        :param start: The starting index for the frames.
        :param end: The ending index for the frames.
        :param step: The step for the frames.

        :return: The loaded file data.
        """

        if silent is None:
            silent = self.silent

        path = path or self.source

        if path is None:
            raise ValueError("No path specified.")

        path = str(path)

        self._audio = AudioFileClip(path)

        start = start or 0
        end = end or self.length or int(
            self._audio.duration * self._audio.fps
        )
        step = step or 1

        iterations = range(start, end, step)

        iterations = tqdm(
            iterations,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{remaining}s, {rate_fmt}{postfix}]"
            ),
            desc=f"Loading audio from {Path(path)}",
            total=len(iterations)
        ) if not silent else iterations

        values = np.arange(0, end, 1.0 / self.fps)

        frames = []

        for i in iterations:
            frame = self._audio.get_frame(values[i])
            frames.append(frame)

            yield frame

        self._audio.reader.close_proc()

        self.source = path

    def load_frames(
            self,
            path: str | Path,
            silent: bool = None,
            start: int = None,
            end: int = None,
            step: int = None
    ) -> None:
        """
        Loads the video data from the file.

        :param path: The path to the source file.
        :param silent: The value for no output.
        :param start: The starting index for the frames.
        :param end: The ending index for the frames.
        :param step: The step for the frames.
        """

        self.frames.extend(
            self.load_frames_generator(
                path=path, silent=silent,
                start=start, end=end, step=step
            )
        )

        self._update_audio()

    @classmethod
    def load(
            cls,
            path: str | Path,
            destination: str | Path = None,
            silent: bool = None,
            frames: Iterable[np.ndarray] = None,
            start: int = None,
            end: int = None,
            step: int = None
    ) -> Self:
        """
        Loads the data from the file.

        :param path: The path to the source file.
        :param destination: The destination to set for the video object.
        :param silent: The value for no output.
        :param frames: The frames to insert to the video data object.
        :param start: The starting index for the frames.
        :param end: The ending index for the frames.
        :param step: The step for the frames.

        :return: The loaded file data.
        """

        path = str(path)

        cap = cv2.VideoCapture(path)

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = list(frames or [])

        audio = cls(
            frames=frames, fps=fps,
            source=path, destination=destination
        )

        audio.load_frames(
            path=path, silent=silent,
            start=start, end=end or length, step=step
        )

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

        codec = 'pcm_s16le' if path.endswith(".avi") else None

        if location := Path(path).parent:
            os.makedirs(location, exist_ok=True)

        self._audio.write_audiofile(
            path, fps=44100, codec=codec, verbose=False, logger=None
        )

        self.destination = path

    def copy(self) -> Self:
        """Creates a copy of the data."""

        audio = Audio(
            frames=[frame.copy() for frame in self.frames],
            fps=self.fps, source=self.source,
            destination=self.destination,
            silent=self.silent,
            audio=self._audio.copy()
        )

        audio._update_audio()

        return audio

    def inherit(self, audio: Self) -> None:
        """
        Inherits the data from the given audio.

        :param audio: The source audio object.
        """

        self.frames = [frame.copy() for frame in audio.frames]
        self._audio = audio._audio.copy()

        self.source = audio.source
        self.destination = audio.destination
        self.silent = audio.silent

        self._update_audio()
