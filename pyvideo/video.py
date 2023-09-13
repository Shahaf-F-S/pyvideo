# video.py

from typing import (
    Optional, List, Union, Tuple,
    Generator, Iterable, Any, ClassVar
)
from pathlib import Path

from attrs import define, field

import numpy as np
import cv2
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip, AudioFileClip

__all__ = [
    "Video"
]

@define(hash=True, slots=False)
class Video:
    """A class to contain video metadata."""

    fps: float

    width: Optional[int] = None
    height: Optional[int] = None

    source: Optional[Union[str, Path]] = None
    destination: Optional[Union[str, Path]] = None

    silent: Optional[bool] = True

    frames: Optional[List[np.ndarray]] = field(factory=list)
    audio: Optional[List[np.ndarray]] = field(factory=list)

    SILENT: ClassVar[bool] = True

    try:
        from typing import Self

    except ImportError:
        Self = Any
    # end try

    def __attrs_post_init__(self) -> None:
        """Defines the attributes of the class."""

        if self.silent is None:
            self.silent = self.SILENT
        # end if

        if self.frames is None:
            self.frames = []
        # end if

        if self.frames:
            if self.width is None:
                self.width = self.frames[0].shape[1]
            # end if

            if self.height is None:
                self.height = self.frames[0].shape[0]
            # end if
        # end if

        self._audio: Optional[AudioFileClip] = None
    # end __attrs_post_init__

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

        video.frames[:] = video.frames[start:end:step]
        video.audio[:] = video.audio[start:end:step]

        return video
    # end cut

    def fit(self, inplace: Optional[bool] = False) -> Self:
        """
        Resizes the frames in the video.

        :param inplace: The value to set changes in the object.

        :return: The modified video object.
        """

        return self.resize(size=(self.width, self.height), inplace=inplace)
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

        video.frames[:] = [cv2.resize(frame, size) for frame in video.frames]
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

    def load_audio_generator(
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
            desc=f"Loading video audio from {Path(path)}",
            total=len(iterations)
        ) if not silent else iterations

        # noinspection PyAttributeOutsideInit
        self._audio = AudioFileClip(path)

        values = np.arange(0, end, 1.0 / self.fps)

        frames = []

        def make_frame(t: Union[float, np.ndarray]) -> Union[np.ndarray, Iterable[np.ndarray]]:
            """
            Returns the frame or frames of audio for the given time.

            :param t: The time of the audio.

            :return: The frame or frames of audio.
            """

            _frames = np.array(frames)

            if isinstance(t, np.ndarray):
                array_indexes = (self.fps*t).astype(int)
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

        for i in iterations:
            frame = self._audio.get_frame(values[i])
            frames.append(frame)

            yield frame
        # end for

        self._audio.reader.close_proc()
        self._audio.reader.make_frame = make_frame
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
                path=path, silent=silent, start=start, end=end, step=step
            )
        )
    # end load_frames

    def load_audio(
            self,
            path: Union[str, Path],
            silent: Optional[bool] = None,
            start: Optional[int] = None,
            end: Optional[int] = None,
            step: Optional[int] = None
    ) -> None:
        """
        Loads the audio data from the file.

        :param path: The path to the source file.
        :param silent: The value for no output.
        :param start: The starting index for the frames.
        :param end: The ending index for the frames.
        :param step: The step for the frames.
        """

        self.audio.extend(
            self.load_audio_generator(
                path=path, silent=silent, start=start, end=end, step=step
            )
        )
    # end load_audio

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

        video = cls(
            frames=frames,
            fps=fps, width=width, height=height,
            source=path, destination=destination
        )

        video.load_audio(
            path=path, silent=silent,
            start=start, end=end or length, step=step
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
            start: Optional[int] = None,
            end: Optional[int] = None,
            step: Optional[int] = None,
            audio: Optional[bool] = None,
            video: Optional[bool] = None
    ) -> None:
        """
        Saves the video and audio into the file.

        :param path: The saving path.
        :param start: The starting index for the frames.
        :param end: The ending index for the frames.
        :param step: The step for the frames.
        :param audio: The value to save the audio.
        :param video: The value to save the frames.
        """

        if audio is None:
            audio = True
        # end if

        if video is None:
            video = True
        # end if

        path = path or self.destination

        if path is None:
            raise ValueError("No path specified.")
        # end if

        path = str(path)

        start = start or 0
        end = end or self.length
        step = step or 1

        audio_clip = None

        if audio:
            audio_clip = self._audio
        # end if

        if video:
            video_clip = ImageSequenceClip(self.frames[start:end:step], fps=self.fps)
            video_clip = video_clip.set_audio(audio_clip)
            video_clip.write_videofile(path, fps=self.fps, verbose=False, logger=None)
            video_clip.close()
        # end if

        if audio and not video:
            codec = 'pcm_s16le' if path.endswith(".avi") else None
            audio_clip.write_audiofile(
                path, fps=self.fps, verbose=False, logger=None, codec=codec
            )
        # end if

        if audio:
            self._audio.reader.close_proc()
        # end if
    # end _save

    def save_frames(
            self,
            path: Optional[Union[str, Path]] = None,
            start: Optional[int] = None,
            end: Optional[int] = None,
            step: Optional[int] = None
    ) -> None:
        """
        Saves the video and audio into the file.

        :param path: The saving path.
        :param start: The starting index for the frames.
        :param end: The ending index for the frames.
        :param step: The step for the frames.
        """

        self._save(path=path, start=start, end=end, step=step, audio=False)
    # end save

    def save_audio(
            self,
            path: Optional[Union[str, Path]] = None,
            start: Optional[int] = None,
            end: Optional[int] = None,
            step: Optional[int] = None
    ) -> None:
        """
        Saves the video and audio into the file.

        :param path: The saving path.
        :param start: The starting index for the frames.
        :param end: The ending index for the frames.
        :param step: The step for the frames.
        """

        self._save(path=path, start=start, end=end, step=step, video=False)
    # end save_audio

    def save(
            self,
            path: Optional[Union[str, Path]] = None,
            start: Optional[int] = None,
            end: Optional[int] = None,
            step: Optional[int] = None
    ) -> None:
        """
        Saves the video and audio into the file.

        :param path: The saving path.
        :param start: The starting index for the frames.
        :param end: The ending index for the frames.
        :param step: The step for the frames.
        """

        self._save(path=path, start=start, end=end, step=step)
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

        video._audio = self._audio.copy()

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

        # noinspection PyAttributeOutsideInit
        self._audio = video._audio.copy()
    # end inherit
# end Video