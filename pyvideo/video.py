# video.py

from typing import (
    Optional, List, Union, Tuple, Self, Generator, Iterable
)
from pathlib import Path

from attrs import define

import numpy as np
import cv2
from tqdm import tqdm

__all__ = [
    "Video"
]

@define(hash=True)
class Video:
    """A class to contain video metadata."""

    fps: float
    width: int
    height: int

    frames: List[np.ndarray]
    source: Optional[Union[str, Path]] = None
    destination: Optional[Union[str, Path]] = None

    @property
    def length(self) -> int:
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

    def load_frames_generator(
            self,
            path: Optional[Union[str, Path]] = None,
            silent: Optional[bool] = None
    ) -> Generator[np.ndarray, None, None]:
        """
        Loads the data from the file.

        :param path: The path to the source file.
        :param silent: The value for no output.

        :return: The loaded file data.
        """

        path = path or self.source

        if path is None:
            raise ValueError("No path specified.")
        # end if

        path = str(path)

        cap = cv2.VideoCapture(path)

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        retrieve = True

        iterations = range(length)

        iterations = tqdm(
            iterations,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{remaining}s, {rate_fmt}{postfix}]"
            ),
            desc=f"Loading video frames from {Path(path)}",
            total=length
        ) if not silent else iterations

        new_source = (
            (self.source is not None) and
            (path is not None) and
            (self.source != path)
        )

        if new_source:
            self.frames.clear()
        # end if

        if (
            (len(self.frames) > 0) and
            not new_source
        ):
            cap.set(cv2.CAP_PROP_POS_FRAMES, len(self.frames))

            iterations -= len(self.frames)
        # end if

        for _ in iterations:
            if not (cap.isOpened() and retrieve):
                break
            # end if

            retrieve, frame = cap.read()

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self.frames.append(frame)

            yield frame
        # end for
    # end load_frames_generator

    def load_frames(
            self,
            path: Union[str, Path],
            silent: Optional[bool] = None
    ) -> None:
        """
        Loads the data from the file.

        :param path: The path to the source file.
        :param silent: The value for no output.
        """

        for _ in self.load_frames_generator(path=path, silent=silent):
            pass
        # end for
    # end load_frames

    @classmethod
    def load(
            cls,
            path: Union[str, Path],
            silent: Optional[bool] = False,
            frames: Optional[Iterable[np.ndarray]] = None,
            load_frames: Optional[bool] = True
    ) -> Self:
        """
        Loads the data from the file.

        :param path: The path to the source file.
        :param silent: The value for no output.
        :param frames: The frames to insert to the video data object.
        :param load_frames: The value to load_frames the frames.

        :return: The loaded file data.
        """

        path = str(path)

        cap = cv2.VideoCapture(path)

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frames is None:
            frames = []
        # end if

        frames = list(frames)

        data = cls(
            frames=frames,
            fps=fps,
            width=width,
            height=height
        )

        if load_frames:
            data.load_frames(path=path, silent=silent)
        # end if

        return data
    # end load

    def save_frames_generator(
            self,
            path: Optional[Union[str, Path]] = None,
            silent: Optional[bool] = None
    ) -> Generator[np.ndarray, None, None]:
        """
        Saves the video and audio into the file.

        :param path: The saving path.
        :param silent: The value for no output.
        """

        path = path or self.destination

        if path is None:
            raise ValueError("No path specified.")
        # end if

        path = str(path)

        length = len(self.frames)

        iterations = range(length)

        iterations = tqdm(
            iterations,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{remaining}s, {rate_fmt}{postfix}]"
            ),
            desc=f"Saving video data to {Path(path)}",
            total=length
        ) if not silent else iterations

        if not silent:
            print()
        # end if

        result = cv2.VideoWriter(path, -1, self.fps, self.size)

        if not silent:
            print()
        # end if

        for i in iterations:
            result.write(self.frames[i])

            yield self.frames[i]
        # end for

        result.release()
    # end save

    def save_frames(
            self,
            path: Optional[Union[str, Path]] = None,
            silent: Optional[bool] = None
    ) -> None:
        """
        Saves the video and audio into the file.

        :param path: The saving path.
        :param silent: The value for no output.
        """

        for _ in self.save_frames_generator(path=path, silent=silent):
            pass
        # end for
    # end save_frames

    def save(
            self,
            path: Optional[Union[str, Path]] = None,
            silent: Optional[bool] = None
    ) -> None:
        """
        Saves the video and audio into the file.

        :param path: The saving path.
        :param silent: The value for no output.
        """

        self.save_frames(path=path, silent=silent)
    # end save

    def copy(self) -> Self:
        """Creates a copy of the data."""

        return Video(
            frames=self.frames,
            fps=self.fps,
            width=self.width,
            height=self.height
        )
    # end copy
# end Video