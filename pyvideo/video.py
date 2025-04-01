# video.py

import os
from typing import Generator, Iterable, Self
from pathlib import Path
from abc import ABCMeta
from dataclasses import dataclass

import numpy as np
import cv2
from moviepy import ImageSequenceClip

from pyvideo.audio import BaseAudio, AudioArray, AudioList
from pyvideo.base import TimedFrames, TimedFramesList, TimedFramesArray


__all__ = [
    "BaseVideo",
    "VideoList",
    "VideoArray"
]


@dataclass
class BaseVideo(TimedFrames, metaclass=ABCMeta):
    """A class to contain video metadata."""

    audio: BaseAudio = None

    def __post_init__(self):
        if self.audio is not None:
            self.set_audio(self.audio)

    @property
    def width(self) -> int:
        return self.frames[0].shape[1]

    @property
    def height(self) -> int:
        return self.frames[0].shape[0]

    @property
    def size(self) -> tuple[int, int]:
        """
        Returns the size of each frame in the video.

        :return: The tuple for x and y values.
        """

        return self.width, self.height

    @property
    def shape(self) -> tuple[int, int]:
        """
        Returns the shape of each frame in the video.

        :return: The tuple for x and y values.
        """

        return self.height, self.width

    @property
    def aspect_ratio(self) -> float:
        """
        Returns the aspect ratio each frame in the video.

        :return: The aspect ratio.
        """

        return self.width / self.height

    def set_audio(self, audio: BaseAudio) -> Self:
        if self.audio is not None:
            self.children.remove(self.audio)

        self.audio = audio

        self.children.append(self.audio)

    def resize(self, size: tuple[int, int]) -> Self:
        """
        Resizes the frames in the video.

        :param size: The new size of the frames.

        :return: The modified video object.
        """




        blob = cv2.dnn.blobFromImages(
            np.array(self.frames),
            scalefactor=1.0, size=size, swapRB=False, crop=False
        )
        self.frames[:] = blob.transpose(0, 2, 3, 1)

        return self

    def reshape(self, size: tuple[int, int]) -> Self:
        """
        Resizes the frames in the video.

        :param size: The new size of the frames.

        :return: The modified video object.
        """




        self.frames[:] = np.array(self.frames).reshape((-1, *size, 3))

        return self

    def rescale(self, factor: float) -> Self:
        """
        Resizes the frames in the video.

        :param factor: The new size of the frames.

        :return: The modified video object.
        """




        size = (int(self.width * factor), int(self.height * factor))

        return self.resize(size=size)

    def crop(
        self,
        upper_left: tuple[int, int],
        lower_right: tuple[int, int]
    ) -> Self:
        """
        Crops the frames of the video.

        :param upper_left: The index of the upper left corner.
        :param lower_right: The index of the lower right corner.

        :return: The modified video object.
        """




        width = lower_right[0] - upper_left[0]
        height = lower_right[1] - upper_left[1]

        if self.frames:
            if (
                (width > self.width) or
                (height > self.height) or
                not (0 <= lower_right[0] <= self.width)
                or not (0 <= upper_left[1] <= self.height)
            ):
                raise ValueError(
                    f"Combination of upper left corner: {upper_left} "
                    f"and lowe right corner: {lower_right} is invalid "
                    f"for frames of shape: {self.size}"
                )

            self.frames[:] = [
                frame[
                    upper_left[1]:lower_right[1],
                    upper_left[0]:lower_right[0]
                ] for frame in self.frames
            ]

        return self

    def color(
        self,
        contrast: float = None,
        brightness: float = None
    ) -> Self:
        """
        Edits the color of the frames.

        :param contrast: The contrast factor.
        :param brightness: The brightness factor.

        :return: The modified video object.
        """




        contrast = 1 if contrast is None else contrast
        brightness = 1 if brightness is None else brightness

        beta = int((brightness - 1) * 100)
        self.frames[:] = np.clip(
            np.array(self.frames) * contrast + beta, 0, 255
        ).astype(np.uint8)

        return self

    def flip(
        self,
        vertically: bool = None,
        horizontally: bool = None
    ) -> Self:
        """
        Flips the frames.

        :param vertically: The value to flip the frames vertically.
        :param horizontally: The value to flip the frames horizontally.

        :return: The modified video object.
        """




        if not (horizontally or vertically):
            return self

        frames = np.array(self.frames)

        if vertically:
            frames = frames[:, ::-1, :, :]

        if horizontally:
            frames = frames[:, :, ::-1, :]

        self.frames[:] = frames

        return self

    def read_frames(
        self,
        path: str | Path,
        start: int = None,
        end: int = None,
        step: int = None
    ) -> Generator[np.ndarray, None, None]:
        """
        Loads the frames data from the file.

        :param path: The path to the source file.
        :param start: The starting index for the frames.
        :param end: The ending index for the frames.
        :param step: The step for the frames.

        :return: The loaded file data.
        """

        start = start or 0
        end = end or self.length
        step = step or 1

        cap = cv2.VideoCapture(str(path))

        if start != 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        for i in range(start, end, step):
            if i == end:
                break

            if step != 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)

            _, frame = cap.read()

            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cap.release()

    def load_frames(
        self,
        path: str | Path,
        start: int = None,
        end: int = None,
        step: int = None
    ) -> None:
        """
        Loads the video data from the file.

        :param path: The path to the source file.
        :param start: The starting index for the frames.
        :param end: The ending index for the frames.
        :param step: The step for the frames.
        """

        self.set_frames(
            list(self.read_frames(path=path, start=start, end=end, step=step))
        )

    def load_audio(
        self,
        path: str | Path,
        start: int = None,
        end: int = None,
        step: int = None,
        chunk_size: int | None = 50000
    ) -> None:
        """
        Loads the audio data from the file.

        :param path: The path to the source file.
        :param start: The starting index for the frames.
        :param end: The ending index for the frames.
        :param step: The step for the frames.
        :param chunk_size: The chunk size of each read.
        """

        self.audio = self.audio_type().load(path=path, chunk_size=chunk_size)

        self.cut_child(self.audio, start=start, end=end, step=step)

    @classmethod
    def load(
        cls,
        path: str | Path,
        frames: Iterable[np.ndarray] = None,
        audio: bool | BaseAudio = True,
        chunk_size: int | None = 50000,
        start: int = None,
        end: int = None,
        step: int = None
    ) -> Self:
        """
        Loads the data from the file.

        :param path: The path to the source file.
        :param frames: The frames to insert to the video data object.
        :param audio: The audio object or the value to load the audio object.
        :param chunk_size: The chunk size of each read.
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

        video = cls(
            frames=frames,
            audio=(audio if not isinstance(audio, bool) else None),
            fps=fps
        )

        video.load_frames(
            path=path, start=start, end=end or length, step=step
        )

        if audio is True:
            video.load_audio(
                path=path, start=start, end=end, step=step,
                chunk_size=chunk_size
            )

        return video

    def save(
        self,
        path: str | Path,
        audio: bool | BaseAudio = None
    ) -> None:
        """
        Saves the video and audio into the file.

        :param path: The saving path.
        :param audio: The value to save the audio.
        """

        audio: BaseAudio | bool

        if audio is None and isinstance(self.audio, BaseAudio):
            audio = True

        path = str(path)

        video_clip = ImageSequenceClip(self.frames, fps=self.fps)

        if audio is True:
            if self.audio is None:
                raise ValueError(
                    "Audio object is not defined. Make sure audio "
                    "data is loaded before attempting to save it."
                )

            audio: BaseAudio = self.audio

        video_clip.audio = audio.moviepy()

        if location := Path(path).parent:
            os.makedirs(location, exist_ok=True)

        video_clip.write_videofile(
            path, fps=self.fps, codec="libx264", audio_codec="aac", logger=None
        )
        video_clip.close()
        video_clip.audio.close()

    def save_frames(self, path: str | Path) -> None:
        """
        Saves the video and audio into the file.

        :param path: The saving path.
        """

        self.save(path=path, audio=False)

    def audio_type(self) -> type[BaseAudio]:
        return BaseAudio


@dataclass
class VideoList(BaseVideo, TimedFramesList):
    """A class to contain video metadata."""

    def audio_type(self) -> type[AudioList]:
        return AudioList


@dataclass
class VideoArray(BaseVideo, TimedFramesArray):
    """A class to contain video metadata."""

    def audio_type(self) -> type[AudioArray]:
        return AudioArray
