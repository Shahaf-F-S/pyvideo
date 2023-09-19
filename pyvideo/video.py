# video.py

from typing import (
    Optional, List, Union, Tuple,
    Generator, Iterable, Any, ClassVar
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

    try:
        from typing import Self

    except ImportError:
        Self = Any
    # end try

    def __init__(
            self,
            fps: float,
            width: Optional[int] = None,
            height: Optional[int] = None,
            source: Optional[Union[str, Path]] = None,
            destination: Optional[Union[str, Path]] = None,
            silent: Optional[bool] = True,
            frames: Optional[List[np.ndarray]] = None,
            audio: Optional[Audio] = None,
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
        # end if

        if frames is None:
            frames = []
        # end if

        self._fps = fps
        self.width = width
        self.height = height

        self.source = source
        self.destination = destination

        self.silent = silent

        self.frames = frames
        self.audio = audio

        if self.frames:
            if self.width is None:
                self.width = self.frames[0].shape[1]
            # end if

            if self.height is None:
                self.height = self.frames[0].shape[0]
            # end if
        # end if
    # end __init__

    @property
    def length(self) -> float:
        """
        Returns the amount of frames in the video.

        :return: The int amount of frames.
        """

        return len(self.frames)
    # end length

    @property
    def fps(self) -> float:
        """
        Returns the frame per second rate of the video.

        :return: The video speed.
        """

        return self._fps
    # end fps

    @fps.setter
    def fps(self, value: float) -> None:
        """
        Returns the frame per second rate of the video.

        :param value: The video speed.
        """

        self._fps = value

        if isinstance(self.audio, Audio):
            self.audio.fps = value
        # end if
    # end fps

    @property
    def duration(self) -> float:
        """
        Returns the amount of time in the video.

        :return: The int amount of time.
        """

        return self.fps * self.length
    # end duration

    @property
    def size(self) -> Tuple[int, int]:
        """
        Returns the size of each frame in the video.

        :return: The tuple for x and y values.
        """

        return self.width, self.height
    # end size

    @property
    def aspect_ratio(self) -> float:
        """
        Returns the aspect ratio each frame in the video.

        :return: The aspect ratio.
        """

        return self.width / self.height
    # end size

    def time_frame(self) -> List[float]:
        """
        Returns a list of the time points.

        :return: The list of time points.
        """

        return [
            i * (self.duration / len(self.frames))
            for i in range(1, len(self.frames) + 1)
        ]
    # end time_frame

    def cut(
            self,
            start: Optional[int] = None,
            end: Optional[int] = None,
            step: Optional[int] = None,
            inplace: Optional[bool] = False
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
        # end if

        start = start or 0
        end = end or self.length
        step = step or 1

        if video.frames:
            video.frames[:] = video.frames[start:end:step]
        # end if

        video.audio = video.audio.cut(
            start=start, end=end, step=step, inplace=inplace
        )

        return video
    # end cut

    def fit(self, inplace: Optional[bool] = False) -> Self:
        """
        Resizes the frames in the video.

        :param inplace: The value to set changes in the object.

        :return: The modified video object.
        """

        return self.resize(
            size=(self.width, self.height), inplace=inplace
        )
    # end fit

    def resize(self, size: Tuple[int, int], inplace: Optional[bool] = False) -> Self:
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
        # end if

        video.frames[:] = [
            cv2.resize(frame, size) for frame in video.frames
        ]
        video.width = size[0]
        video.height = size[1]

        return video
    # end resize

    def rescale(self, factor: float, inplace: Optional[bool] = False) -> Self:
        """
        Resizes the frames in the video.

        :param factor: The new size of the frames.
        :param inplace: The value to set changes in the object.

        :return: The modified video object.
        """

        size = (int(self.width * factor), int(self.height * factor))

        return self.resize(size=size, inplace=inplace)
    # end rescale

    def crop(
            self,
            upper_left: Tuple[int, int],
            lower_right: Tuple[int, int],
            inplace: Optional[bool] = False
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
        # end if

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
            # end if

            video.frames[:] = [
                frame[
                    upper_left[1]:lower_right[1],
                    upper_left[0]:lower_right[0]
                ] for frame in video.frames
            ]
        # end if

        video.width = width
        video.height = height

        return video
    # end crop

    def color(
            self,
            contrast: Optional[float] = None,
            brightness: Optional[float] = None,
            inplace: Optional[bool] = False
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
        # end if

        if brightness is None:
            brightness = 1
        # end if

        if inplace:
            video = self

        else:
            video = self.copy()
        # end if

        video.frames[:] = [
            cv2.convertScaleAbs(
                frame, alpha=contrast, beta=int((brightness - 1) * 100)
            ) for frame in video.frames
        ]

        return video
    # end color

    def load_frames_generator(
            self,
            path: Optional[Union[str, Path]] = None,
            silent: Optional[bool] = None,
            start: Optional[int] = None,
            end: Optional[int] = None,
            step: Optional[int] = None
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
        # end if

        path = path or self.source

        if path is None:
            raise ValueError("No path specified.")
        # end if

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
        # end if

        for i in iterations:
            if i == end:
                break
            # end if

            if step != 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            # end if

            _, frame = cap.read()

            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # end for

        cap.release()
    # end load_frames_generator

    def load_frames(
            self,
            path: Union[str, Path],
            silent: Optional[bool] = None,
            start: Optional[int] = None,
            end: Optional[int] = None,
            step: Optional[int] = None
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
                path=path, silent=silent, start=start, end=end, step=step
            )
        )
    # end load_frames

    @classmethod
    def load(
            cls,
            path: Union[str, Path],
            destination: Optional[Union[str, Path]] = None,
            silent: Optional[bool] = None,
            frames: Optional[Iterable[np.ndarray]] = None,
            start: Optional[int] = None,
            end: Optional[int] = None,
            step: Optional[int] = None
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
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frames is None:
            frames = []
        # end if

        frames = list(frames)

        audio = Audio.load(
            path=path, silent=silent, start=start, end=end, step=step
        )

        video = cls(
            frames=frames, audio=audio,
            fps=fps, width=width, height=height,
            source=path, destination=destination
        )

        video.load_frames(
            path=path, silent=silent,
            start=start, end=end or length, step=step
        )

        return video
    # end load

    def _save(
            self,
            path: Optional[Union[str, Path]] = None,
            audio: Optional[bool] = None
    ) -> None:
        """
        Saves the video and audio into the file.

        :param path: The saving path.
        :param audio: The value to save the audio.
        """

        if audio is None:
            audio = True
        # end if

        path = path or self.destination

        if path is None:
            raise ValueError("No path specified.")
        # end if

        path = str(path)

        audio_clip = None

        if audio:
            # noinspection PyProtectedMember
            audio_clip = self.audio._audio
        # end if

        video_clip = ImageSequenceClip(self.frames, fps=self.fps)
        video_clip = video_clip.set_audio(audio_clip)
        video_clip.write_videofile(path, fps=self.fps, verbose=False, logger=None)
        video_clip.close()

        if audio:
            # noinspection PyProtectedMember
            self.audio._audio.reader.close_proc()
        # end if
    # end _save

    def save_frames(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Saves the video and audio into the file.

        :param path: The saving path.
        """

        self._save(path=path, audio=False)
    # end save

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Saves the video and audio into the file.

        :param path: The saving path.
        """

        self._save(path=path, audio=True)
    # end save

    def copy(self) -> Self:
        """Creates a copy of the data."""

        video = Video(
            frames=self.frames.copy(),
            audio=self.audio.copy(),
            fps=self.fps, width=self.width, height=self.height,
            source=self.source, destination=self.destination,
            silent=self.silent
        )

        return video
    # end copy

    def inherit(self, video: Self) -> None:
        """
        Inherits the data from the given video.

        :param video: The source video object.
        """

        self.frames = video.frames.copy()
        self.audio = video.audio.copy()

        self.fps = video.fps
        self.width = video.width
        self.height = video.height
        self.source = video.source
        self.destination = video.destination
        self.silent = video.silent
    # end inherit
# end Video