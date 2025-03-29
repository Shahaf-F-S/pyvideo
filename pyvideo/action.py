# action.py

from dataclasses import dataclass, field
from typing import Self, Generator, Callable, Iterable

import numpy as np


__all__ = [
    "TimedFrames"
]


@dataclass
class TimedFrames:

    fps: float
    frames: list[np.ndarray] = field(default_factory=list)
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

    def cut(
        self,
        start: int = None,
        end: int = None,
        step: int = None
    ) -> Self:
        """
        Cuts the video.

        :param start: The starting index for the frames.
        :param end: The ending index for the frames.
        :param step: The step for the frames.
        :return: The modified video object.
        """

        start = start or 0
        end = end or self.length
        step = step or 1

        self.frames[:] = self.frames[start:end:step]

        for child in self.children:
            self.cut_child(child)

        return self

    def cut_child(
        self,
        child: "TimedFrames",
        start: int = None,
        end: int = None,
        step: int = None
    ) -> "TimedFrames":
        """
        Cuts the video.

        :param child: The child object to cut according to the indexes.
        :param start: The starting index for the frames.
        :param end: The ending index for the frames.
        :param step: The step for the frames.
        :return: The modified video object.
        """
        child_start = child.index(self.time(start))

        child_end = None
        child_step = None

        if end != self.length:
            child_end = child.index(self.time(end))

        if step is not None:
            child_step = int(step * (1 / self.fps) * child.fps)

        return child.cut(start=child_start, end=child_end, step=child_step)

    def speed(self, factor: float) -> Self:
        """
        Changes the speed of the playing.

        :param factor: The speed factor.

        :return: The changes audio object.
        """

        self.fps *= factor

        for child in self.children:
            child.speed(factor)

        return self

    def array(self) -> np.ndarray:
        return np.array(self.frames)


@dataclass
class Action[I, O]:

    data: "TimedFrames | Action[..., I]"
    transform: Callable[[I], O]
    extractor: Callable[[object], Iterable[I]] = None

    def __call__(self) -> Generator[O, ..., ...]:
        data = self.data

        if isinstance(data, Action):
            data = data()

        if self.extractor is not None:
            data = self.extractor(data)

        for value in data:
            yield self.transform(value)

#
# @dataclass
# class Frames(Action[TimedFrames, np.ndarray]):
#
#     data: TimedFrames
#     deep: bool = False
#     transform = lambda obj: obj.array()
#
#     def __post_init__(self):
#         if self.extractor is None:
#             self.extractor = lambda data: (
#                 data, *(data.children if self.deep else ())
#             )
#
# @dataclass
# class Frame(Action[np.ndarray, np.ndarray]):
#
#     pass