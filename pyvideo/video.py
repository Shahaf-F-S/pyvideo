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
from pyvideo.base import TimedFrames, TimedFramesList, TimedFramesArray, action


__all__ = [
    "BaseVideo",
    "VideoList",
    "VideoArray",
    "resize",
    "reshape",
    "rescale",
    "crop",
    "color",
    "flip"
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
        if (self.audio is not None) and (self.audio in self.children):
            self.children.remove(self.audio)

        self.audio = audio

        self.children.append(self.audio)

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

        audio = self.audio_type().load(path=path, chunk_size=chunk_size)
        self.cut_child(audio, start=start, end=end, step=step)
        self.set_audio(audio)

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

        video_clip = ImageSequenceClip(self.list(), fps=self.fps)

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


def resize[T: BaseVideo](
    video: T, size: tuple[int, int], deep: int | bool = True
) -> Iterable[None | np.ndarray]:
    yield

    if isinstance(video, VideoArray):
        blob = cv2.dnn.blobFromImages(
            np.array(video.frames),
            scalefactor=1.0, size=size, swapRB=False, crop=False
        )
        frames = blob.transpose(0, 2, 3, 1)
        video.frames = frames

        yield frames

    else:
        for i, frame in enumerate(video.frames):
            frame = cv2.resize(frame, size)
            yield frame
            video.frames[i] = frame

    if deep:
        deep = deep - 1 if deep is not True else deep

        for child in video.children:
            if not isinstance(child, BaseVideo):
                continue

            yield from resize(child, size=size, deep=deep)


def reshape[T: BaseVideo](
    video: T, size: tuple[int, int], deep: int | bool = True
) -> Iterable[None | np.ndarray]:
    yield

    if isinstance(video, VideoArray):
        frames = video.frames.reshape(-1, *size, 3)
        video.frames[:] = frames

        yield frames

    else:
        for i, frame in enumerate(video.frames):
            frame = frame.reshape(*size, 3)
            yield frame
            video.frames[i] = frame

    if deep:
        deep = deep - 1 if deep is not True else deep

        for child in video.children:
            if not isinstance(child, BaseVideo):
                continue

            yield from reshape(child, size=size, deep=deep)


def rescale[T: BaseVideo](
    video: T, factor: float | tuple[float, float], deep: int | bool = True
) -> Iterable[None | np.ndarray]:
    yield

    if not isinstance(factor, tuple):
        f = factor
        factor = (f, f)

    factor1, factor2 = factor

    # noinspection PyTypeChecker
    size = (int(video.width * factor1), int(video.height * factor2))

    yield from resize(video, size=size, deep=deep)


def crop[T: BaseVideo](
    video: T,
    upper_left: tuple[int, int],
    lower_right: tuple[int, int],
    deep: int | bool = True
) -> Iterable[None | np.ndarray]:
    yield

    width = lower_right[0] - upper_left[0]
    height = lower_right[1] - upper_left[1]

    if len(video.frames) == 0:
        # noinspection PyTypeChecker
        if (
            (width > video.width) or
            (height > video.height) or
            not (0 <= lower_right[0] <= video.width)
            or not (0 <= upper_left[1] <= video.height)
        ):
            raise ValueError(
                f"Combination of upper left corner: {upper_left} "
                f"and lowe right corner: {lower_right} is invalid "
                f"for frames of shape: {video.size}"
            )

    if isinstance(video, VideoArray):
        # noinspection PyTypeChecker
        frames: np.ndarray = video.frames[
            :,
            upper_left[1]:lower_right[1],
            upper_left[0]:lower_right[0]
        ]

        yield frames

        video.frames = frames

    else:
        for i, frame in enumerate(video.frames):
            frame = frame[
                upper_left[1]:lower_right[1],
                upper_left[0]:lower_right[0]
            ]
            yield frame
            video.frames[i] = frame

    if deep:
        deep = deep - 1 if deep is not True else deep

        for child in video.children:
            if not isinstance(child, BaseVideo):
                continue

            yield from crop(
                child,
                upper_left=upper_left,
                lower_right=lower_right,
                deep=deep
            )


def color[T: BaseVideo](
    video: T,
    contrast: float = None,
    brightness: float = None,
    deep: int | bool = True
) -> Iterable[np.ndarray]:
    contrast = 1 if contrast is None else contrast
    brightness = 1 if brightness is None else brightness
    beta = int((brightness - 1) * 100)

    def change(d: np.ndarray) -> np.ndarray:
        return np.clip(d * contrast + beta, 0, 255).astype(np.uint8)

    yield from action(video, change_data=change, deep=deep, base=BaseVideo)


def flip[T: BaseVideo](
    video: T,
    vertically: bool = None,
    horizontally: bool = None,
    deep: int | bool = True
) -> Iterable[np.ndarray]:

    def change(d: np.ndarray) -> np.ndarray:
        if len(d.shape) > 3:
            if vertically:
                d = d[:, ::-1, :, :]

            if horizontally:
                d = d[:, :, ::-1, :]

        else:
            if vertically:
                d = d[::-1, :, :]

            if horizontally:
                d = d[:, ::-1, :]

        return d

    yield from action(video, change_data=change, deep=deep, base=BaseVideo)
