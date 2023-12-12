# video.py

import os
from typing import (
    Generator, Iterable, Self, ClassVar
)
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip

from pyvideo.audio import Audio

__all__ = [
    "Video"
]

class Video:
    """A class to contain video metadata."""

    SILENT: ClassVar[bool] = True

    def __init__(
            self,
            fps: float,
            width: int = None,
            height: int = None,
            source: str | Path = None,
            destination: str | Path = None,
            silent: bool = True,
            frames: list[np.ndarray] = None,
            audio: Audio = None
    ) -> None:
        """
        Defines the attributes of a video.

        :param fps: The frames per second rate.
        :param width: The width of each frame.
        :param height: The height of each frame.
        :param source: The source file path.
        :param destination: The destination file path.
        :param silent: The value to silent output.
        :param frames: The list of frames.
        :param audio: The list of audio data.
        """

        if silent is None:
            silent = self.SILENT

        if frames is None:
            frames = []

        self._fps = fps
        self.width = width
        self.height = height

        self.source = source
        self.destination = destination

        self.silent = silent

        self.frames = frames
        self._audio = audio

        if self.frames:
            if self.width is None:
                self.width = self.frames[0].shape[1]

            if self.height is None:
                self.height = self.frames[0].shape[0]

    @property
    def length(self) -> float:
        """
        Returns the amount of frames in the video.

        :return: The int amount of frames.
        """

        return len(self.frames)

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

        self._fps = value

        if isinstance(self.audio, Audio):
            self.audio.fps = value

    @property
    def duration(self) -> float:
        """
        Returns the amount of time in the video.

        :return: The int amount of time.
        """

        return round(self.length / self.fps, 12)

    @property
    def size(self) -> tuple[int, int]:
        """
        Returns the size of each frame in the video.

        :return: The tuple for x and y values.
        """

        return self.width, self.height

    @property
    def aspect_ratio(self) -> float:
        """
        Returns the aspect ratio each frame in the video.

        :return: The aspect ratio.
        """

        return self.width / self.height

    @property
    def audio(self) -> Audio:
        """
        Returns the frame per second rate of the video.

        :return: The video speed.
        """

        return self._audio

    @audio.setter
    def audio(self, value: Audio) -> None:
        """
        Returns the frame per second rate of the video.

        :param value: The video speed.
        """

        if any(
            value1 != value2 for value1, value2 in
            zip(
                (self.fps, self.length),
                (value.fps, value.length)
            )
        ):
            raise ValueError(
                "Audio object must have the exact same "
                "identifier values (fps, duration, length) "
                "as the video object"
            )

        self._audio = value

    def time_frame(self) -> list[float]:
        """
        Returns a list of the time points.

        :return: The list of time points.
        """

        return [
            i * (self.duration / len(self.frames))
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
            video = self

        else:
            video = self.copy()

        start = start or 0
        end = end or self.length
        step = step or 1

        if video.frames:
            video.frames[:] = video.frames[start:end:step]

        if isinstance(video.audio, Audio):
            video.audio = video.audio.cut(
                start=start, end=end, step=step, inplace=inplace
            )

        return video

    def fit(self, inplace: bool = False) -> Self:
        """
        Resizes the frames in the video.

        :param inplace: The value to set changes in the object.

        :return: The modified video object.
        """

        return self.resize(
            size=(self.width, self.height), inplace=inplace
        )

    def resize(self, size: tuple[int, int], inplace: bool = False) -> Self:
        """
        Resizes the frames in the video.

        :param size: The new size of the frames.
        :param inplace: The value to set changes in the object.

        :return: The modified video object.
        """

        if inplace:
            video = self

        else:
            video = self.copy()

        video.frames[:] = [
            cv2.resize(frame, size) for frame in video.frames
        ]
        video.width = size[0]
        video.height = size[1]

        return video

    def rescale(self, factor: float, inplace: bool = False) -> Self:
        """
        Resizes the frames in the video.

        :param factor: The new size of the frames.
        :param inplace: The value to set changes in the object.

        :return: The modified video object.
        """

        size = (int(self.width * factor), int(self.height * factor))

        return self.resize(size=size, inplace=inplace)

    def crop(
            self,
            upper_left: tuple[int, int],
            lower_right: tuple[int, int],
            inplace: bool = False
    ) -> Self:
        """
        Crops the frames of the video.

        :param upper_left: The index of the upper left corner.
        :param lower_right: The index of the lower right corner.
        :param inplace: The value to set changes in the object.

        :return: The modified video object.
        """

        if inplace:
            video = self

        else:
            video = self.copy()

        width = lower_right[0] - upper_left[0]
        height = lower_right[1] - upper_left[1]

        if video.frames:
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

            video.frames[:] = [
                frame[
                    upper_left[1]:lower_right[1],
                    upper_left[0]:lower_right[0]
                ] for frame in video.frames
            ]

        video.width = width
        video.height = height

        return video

    def color(
            self,
            contrast: float = None,
            brightness: float = None,
            inplace: bool = False
    ) -> Self:
        """
        Edits the color of the frames.

        :param contrast: The contrast factor.
        :param brightness: The brightness factor.
        :param inplace: The value to set changes in the object.

        :return: The modified video object.
        """

        if contrast is None:
            contrast = 1

        if brightness is None:
            brightness = 1

        if inplace:
            video = self

        else:
            video = self.copy()

        video.frames[:] = [
            cv2.convertScaleAbs(
                frame, alpha=contrast, beta=int((brightness - 1) * 100)
            ) for frame in video.frames
        ]

        return video

    def flip(
            self,
            vertically: bool = None,
            horizontally: bool = None,
            inplace: bool = False
    ) -> Self:
        """
        Flips the frames.

        :param vertically: The value to flip the frames vertically.
        :param horizontally: The value to flip the frames horizontally.
        :param inplace: The value to set changes in the object.

        :return: The modified video object.
        """

        if inplace:
            video = self

        else:
            video = self.copy()

        if vertically:
            video.frames[:] = [
                cv2.flip(frame, 0) for frame in video.frames
            ]

        if horizontally:
            video.frames[:] = [
                cv2.flip(frame, 1) for frame in video.frames
            ]

        return video

    def volume(self, factor: float, inplace: bool = False) -> Self:
        """
        Changes the volume of the audio.

        :param factor: The change value.
        :param inplace: The value to save changes to the object.

        :return: The changes audio object.
        """

        if inplace:
            video = self

        else:
            video = self.copy()

        if video.audio is None:
            raise ValueError("Video has no audio object.")

        video._audio = video.audio.volume(factor=factor, inplace=inplace)

        return video

    def speed(self, factor: float, inplace: bool = False) -> Self:
        """
        Changes the speed of the playing.

        :param factor: The speed factor.
        :param inplace: The value to save changes to the object.

        :return: The changes video object.
        """

        if inplace:
            video = self

        else:
            video = self.copy()

        video.fps *= factor

        return video

    def load_frames_generator(
            self,
            path: str | Path = None,
            silent: bool = None,
            start: int = None,
            end: int = None,
            step: int = None
    ) -> Generator[np.ndarray, None, None]:
        """
        Loads the frames data from the file.

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

        start = start or 0
        end = end or self.length
        step = step or 1

        iterations = range(start, end, step)

        iterations = tqdm(
            iterations,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{remaining}s, {rate_fmt}{postfix}]"
            ),
            desc=f"Loading video frames from {Path(path)}",
            total=len(iterations)
        ) if not silent else iterations

        cap = cv2.VideoCapture(path)

        if start != 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        for i in iterations:
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

    def load_audio(
            self,
            path: str | Path,
            silent: bool = None,
            start: int = None,
            end: int = None,
            step: int = None
    ) -> None:
        """
        Loads the audio data from the file.

        :param path: The path to the source file.
        :param silent: The value for no output.
        :param start: The starting index for the frames.
        :param end: The ending index for the frames.
        :param step: The step for the frames.
        """

        self.audio = Audio.load(
            path=path, silent=silent,
            start=start, end=end, step=step
        )

    @classmethod
    def load(
            cls,
            path: str | Path,
            destination: str | Path = None,
            silent: bool = None,
            frames: Iterable[np.ndarray] = None,
            audio: bool | Audio = None,
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
        :param audio: The audio object or the value to load the audio object.
        :param start: The starting index for the frames.
        :param end: The ending index for the frames.
        :param step: The step for the frames.

        :return: The loaded file data.
        """

        if audio is None:
            audio = True

        path = str(path)

        cap = cv2.VideoCapture(path)

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = list(frames or [])

        video = cls(
            frames=frames,
            audio=(audio if not isinstance(audio, bool) else None),
            fps=fps, width=width, height=height,
            source=path, destination=destination
        )

        video.load_frames(
            path=path, silent=silent,
            start=start, end=end or length, step=step
        )

        if audio is True:
            video.load_audio(
                path=path, silent=silent,
                start=start, end=end or length, step=step
            )

        return video

    def save(
            self,
            path: str | Path = None,
            audio: bool | Audio = None
    ) -> None:
        """
        Saves the video and audio into the file.

        :param path: The saving path.
        :param audio: The value to save the audio.
        """

        audio: Audio | bool

        if audio is None and isinstance(self.audio, Audio):
            audio = True

        path = path or self.destination

        if path is None:
            raise ValueError("No path specified.")

        path = str(path)

        video_clip = ImageSequenceClip(self.frames, fps=self.fps)

        if audio:
            if audio is True:
                if self.audio is None:
                    raise ValueError(
                        "Audio object is not defined. Make sure audio "
                        "data is loaded before attempting to save it."
                    )

                self.audio: Audio

                # noinspection PyProtectedMember
                audio_clip = self.audio._audio

            else:
                audio: Audio

                # noinspection PyProtectedMember
                audio_clip = audio._audio

            video_clip = video_clip.set_audio(audio_clip)

        if location := Path(path).parent:
            os.makedirs(location, exist_ok=True)

        video_clip.write_videofile(
            path, fps=self.fps, verbose=False, logger=None
        )
        video_clip.close()

    def save_frames(self, path: str | Path = None) -> None:
        """
        Saves the video and audio into the file.

        :param path: The saving path.
        """

        self.save(path=path, audio=False)

    def copy(self) -> Self:
        """Creates a copy of the data."""

        video = Video(
            frames=[frame.copy() for frame in self.frames],
            audio=(self.audio.copy() if isinstance(self.audio, Audio) else None),
            fps=self.fps, width=self.width, height=self.height,
            source=self.source, destination=self.destination,
            silent=self.silent
        )

        return video

    def inherit(self, video: Self) -> None:
        """
        Inherits the data from the given video.

        :param video: The source video object.
        """

        self.frames = [frame.copy() for frame in video.frames]

        if (self.audio is None) and (video.audio is not None):
            self.audio = video.audio.copy()

        elif None not in (self.audio, video.audio):
            self.audio.inherit(video.audio)

        self.fps = video.fps
        self.width = video.width
        self.height = video.height
        self.source = video.source
        self.destination = video.destination
        self.silent = video.silent
