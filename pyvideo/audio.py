# audio.py

from pathlib import Path
from typing import ClassVar, Any, Optional, Union, List, Generator, Iterable

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

    try:
        from typing import Self

    except ImportError:
        Self = Any
    # end try

    def __init__(
            self,
            fps: Optional[float] = None,
            source: Optional[Union[str, Path]] = None,
            destination: Optional[Union[str, Path]] = None,
            silent: Optional[bool] = True,
            frames: Optional[List[np.ndarray]] = None,
            audio: Optional[AudioClip] = None
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
        # end if

        if frames is None:
            frames = []
        # end if

        self.fps = fps

        self.source = source
        self.destination = destination

        self.silent = silent

        self.frames = frames

        self._audio: Optional[AudioFileClip] = audio
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
    def duration(self) -> float:
        """
        Returns the amount of time in the video.

        :return: The int amount of time.
        """

        return self._audio.duration
    # end duration

    @duration.setter
    def duration(self, value: float) -> None:
        """
        Returns the amount of time in the video.

        :param value: The int amount of time.
        """

        self._audio.duration = value
    # end duration

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
            audio = self

        else:
            audio = self.copy()
        # end if

        start = start or 0
        end = end or audio.length
        step = step or 1

        if audio.frames:
            audio.frames[:] = audio.frames[start:end:step]
        # end if

        # audio._audio.duration = audio.duration

        return audio
    # end cut

    def _make_frame(self, t: Union[float, np.ndarray]) -> Union[np.ndarray, Iterable[np.ndarray]]:
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
            # end if
        # end if
    # end make_frame

    def _update_audio(self) -> None:
        """Updates the audio data of the object."""

        self._audio.reader.make_frame = self._make_frame
    # end _update_audio

    def load_frames_generator(
            self,
            path: Optional[Union[str, Path]] = None,
            silent: Optional[bool] = None,
            start: Optional[int] = None,
            end: Optional[int] = None,
            step: Optional[int] = None
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
        # end if

        path = path or self.source

        if path is None:
            raise ValueError("No path specified.")
        # end if

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
        # end for

        self.fps = self._audio.fps

        self._audio.reader.close_proc()
    # end load_audio_generator

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
                path=path, silent=silent,
                start=start, end=end, step=step
            )
        )

        self._update_audio()
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
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frames is None:
            frames = []
        # end if

        frames = list(frames)

        audio = cls(
            frames=frames, fps=fps,
            source=path, destination=destination
        )

        audio.load_frames(
            path=path, silent=silent,
            start=start, end=end or length, step=step
        )

        return audio
    # end load

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Saves the video and audio into the file.

        :param path: The saving path
        """

        path = path or self.destination

        if path is None:
            raise ValueError("No path specified.")
        # end if

        path = str(path)

        print(self.fps, self._audio.fps)
        print(self.duration, self._audio.duration)

        codec = 'pcm_s16le' if path.endswith(".avi") else None
        self._audio.write_audiofile(
            path, fps=44100, codec=codec, verbose=False, logger=None
        )

        self._audio.reader.close_proc()
    # end _save

    def copy(self) -> Self:
        """Creates a copy of the data."""

        audio = Audio(
            frames=self.frames.copy(),
            fps=self.fps, source=self.source,
            destination=self.destination,
            silent=self.silent
        )

        audio._audio = self._audio.copy()

        return audio
    # end copy

    def inherit(self, video: Self) -> None:
        """
        Inherits the data from the given video.

        :param video: The source video object.
        """

        self.frames = video.frames.copy()
        self.fps = video.fps
        self.source = video.source
        self.destination = video.destination
        self.silent = video.silent

        self._audio = video._audio.copy()
    # end inherit
# end Audio