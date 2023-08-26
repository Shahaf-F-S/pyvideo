# video.py

from typing import (
    Optional, List, Union, Tuple, Generator, Iterable, Any
)
from pathlib import Path

from attrs import define, field

import numpy as np
import cv2
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip

__all__ = [
    "Video"
]

@define(hash=True)
class Video:
    """A class to contain video metadata."""

    fps: float
    width: int
    height: int
    length: int

    source: Optional[Union[str, Path]] = None
    destination: Optional[Union[str, Path]] = None

    frames: Optional[List[np.ndarray]] = field(factory=list)

    try:
        from typing import Self

    except ImportError:
        Self = Any
    # end try

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

        video.frames[:] = video.frames[start:end:step]
        video.length = len(video.frames)

        return video
    # end cut

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
        video.length = len(video.frames)

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
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Loads the data from the file.

        :param path: The path to the source file.
        :param silent: The value for no output.
        :param start: The starting index for the frames.
        :param end: The ending index for the frames.
        :param step: The step for the frames.

        :return: The loaded file data.
        """

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
        Loads the data from the file.

        :param path: The path to the source file.
        :param silent: The value for no output.
        :param start: The starting index for the frames.
        :param end: The ending index for the frames.
        :param step: The step for the frames.
        """

        for video_frame in self.load_frames_generator(
            path=path, silent=silent, start=start, end=end, step=step
        ):
            self.frames.append(video_frame)
        # end for

        self.length = len(self.frames)
    # end load_frames

    @classmethod
    def load(
            cls,
            path: Union[str, Path],
            destination: Optional[Union[str, Path]] = None,
            silent: Optional[bool] = False,
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

        data = cls(
            frames=frames, length=length,
            fps=fps, width=width, height=height,
            source=path, destination=destination
        )

        data.load_frames(
            path=path, silent=silent,
            start=start, end=end, step=step
        )

        return data
    # end load

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

        path = path or self.destination

        if path is None:
            raise ValueError("No path specified.")
        # end if

        path = str(path)

        video_frames = self.frames[start:end:step]

        clip = ImageSequenceClip(video_frames, fps=self.fps)

        clip.write_videofile(path, fps=self.fps, verbose=False, logger=None)
        clip.close()
    # end save

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

        self.save_frames(path=path, start=start, end=end, step=step)
    # end save

    def copy(self) -> Self:
        """Creates a copy of the data."""

        return Video(
            frames=self.frames.copy(),
            fps=self.fps, width=self.width, height=self.height,
            source=self.source, destination=self.destination,
            length=self.length
        )
    # end copy
# end Video