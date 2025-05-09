# action.py

from dataclasses import dataclass, field, replace
from typing import Self, Iterable, MutableSequence, Callable
from abc import ABCMeta, abstractmethod

import numpy as np


__all__ = [
    "TimedFrames",
    "TimedFramesList",
    "TimedFramesArray",
    "action",
    "speed",
    "cut"
]


@dataclass
class TimedFrames(metaclass=ABCMeta):

    fps: float
    frames: MutableSequence[np.ndarray] = field(default_factory=list, repr=False)
    resolution: int = 12
    children: list["TimedFrames"] = field(default_factory=list)

    def __iter__(self) -> Iterable[np.ndarray]:
        return iter(self.frames)

    def __len__(self) -> int:
        return len(self.frames)

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

    def set_frames(self, frames: MutableSequence[np.ndarray]):
        self.frames = frames

    def speed(self, factor: float, deep: int | bool = True) -> Self:
        self.fps *= factor

        if deep:
            deep = deep - 1 if deep is not True else deep

            for child in self.children:
                child.speed(factor, deep=deep)

        return self

    def sync_duration(self, deep: int | bool = True) -> Self:
        if not deep:
            return self

        for child in self.children:
            deep = deep - 1 if deep is not True else deep

            child.fps *= (child.duration / self.duration)
            child.sync_duration(deep=deep)

        return self

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

    def add(self, child: "TimedFrames"):
        if child is self:
            raise ValueError(f"Cannot add {self} to itself.")

        self.children.append(child)

    def remove(self, child: "TimedFrames"):
        self.children.remove(child)

    def cut(
        self,
        start: int = None,
        end: int = None,
        step: int = None,
        deep: int | bool = True
    ) -> Self:
        """
        Cuts the video.

        :param start: The starting index for the frames.
        :param end: The ending index for the frames.
        :param step: The step for the frames.
        :param deep: The flag or level to perform the action on the children.
        :return: The modified video object.
        """

        self.frames = self.frames[start or 0:end or self.length:step or 1]

        if deep:
            deep = deep - 1 if deep is not True else deep

            for child in self.children:
                self.cut_child(child, start=start, end=end, step=step, deep=deep)

        return self

    def cut_child(
        self,
        child: "TimedFrames",
        start: int = None,
        end: int = None,
        step: int = None,
        deep: int | bool = True
    ) -> "TimedFrames":
        """
        Cuts the video.

        :param child: The child object to cut according to the indexes.
        :param start: The starting index for the frames.
        :param end: The ending index for the frames.
        :param step: The step for the frames.
        :param deep: The flag or level to perform the action on the children.
        :return: The modified video object.
        """

        child_start = None
        child_end = None
        child_step = None

        end = end or self.length

        if start is not None:
            child_start = child.index(self.time(start))

        if end != self.length:
            child_end = child.index(self.time(end))

        if step is not None:
            child_step = int(step * (1 / self.fps) * child.fps)

        child.cut(start=child_start, end=child_end, step=child_step)

        if deep:
            for c in child.children:
                child.cut_child(c, deep=deep)

        return self

    def array(self) -> np.ndarray:
        return np.array(self.frames)

    def list(self) -> list[np.ndarray]:
        return list(self.frames)

    def data(self) -> Iterable[np.ndarray]:
        for frame in self.frames:
            yield frame

    def data_copy(self) -> Iterable[np.ndarray]:
        for frame in self.frames:
            yield frame.copy()

    @abstractmethod
    def copy_frames(self) -> MutableSequence[np.ndarray]:
        """Creates a copy of the data."""
        raise NotImplementedError

    def copy(self) -> Self:
        return replace(
            self,
            frames=self.copy_frames(),
            children=[child.copy() for child in self.children]
        )


@dataclass
class TimedFramesList(TimedFrames):
    """A class to represent data of audio file or audio of video file."""

    frames: list[np.ndarray] = field(default_factory=list, repr=False)

    def copy_frames(self) -> list[np.ndarray]:
        return [frame.copy() for frame in self.frames]

    def set_frames(self, frames: MutableSequence[np.ndarray]):
        self.frames = list(frames)


@dataclass
class TimedFramesArray(TimedFramesList):

    frames: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)

    def data(self) -> Iterable[np.ndarray]:
        yield self.frames

    def copy_frames(self) -> np.ndarray:
        return self.frames.copy()

    def set_frames(self, frames: MutableSequence[np.ndarray]):
        self.frames = np.array(list(frames))


def action[T: TimedFrames](
    obj: T,
    change: Callable[[T], T] = None,
    change_data: Callable[[np.ndarray], np.ndarray] = None,
    deep: int | bool = False,
    base: type[T] = TimedFrames
) -> Iterable[np.ndarray]:
    if not isinstance(obj, base):
        return

    if change:
        obj = change(obj)

    if change_data:
        for frame in obj.data():
            frame[:] = change_data(frame)

            yield frame

    if deep:
        deep = deep - 1 if deep is not True else deep

        for child in obj.children:
            yield from action(child, change_data=change_data, deep=deep, base=base)


def speed[T: TimedFrames](obj: T, factor: float, deep: bool = True) -> Iterable:
    yield

    obj.speed(factor, deep=deep)


def cut[T: TimedFrames](
    obj: T,
    start: int = None,
    end: int = None,
    step: int = None,
    deep: bool = True
) -> Iterable:
    yield

    obj.cut(start=start, end=end, step=step, deep=deep)
