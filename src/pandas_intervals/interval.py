from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, Self


class Time(Protocol):
    """What an interval needs from its time type: ordering and arithmetic."""

    def __lt__(self, other: Self, /) -> bool: ...
    def __le__(self, other: Self, /) -> bool: ...
    def __add__(self, other: Self, /) -> Self: ...
    def __sub__(self, other: Self, /) -> Self: ...
    def __mul__(self, other: Self, /) -> Self: ...


@dataclass(frozen=True, order=True)
class Interval[T: Time]:
    """
    A left-closed interval [left, right) on a time axis of type T.

    T can be any type with ordering and arithmetic, e.g. int, float, or Fraction.
    `left` is the onset and `right` the offset of the interval.
    Intervals are immutable, hashable, and sorted by (left, right).
    """

    left: T
    right: T

    def __post_init__(self) -> None:
        if self.right < self.left:
            raise ValueError(f"right end {self.right} is before left end {self.left}")

    @classmethod
    def from_duration(cls, left: T, duration: T) -> "Interval[T]":
        return cls(left, left + duration)

    @property
    def duration(self) -> T:
        return self.right - self.left

    def __contains__(self, t: T) -> bool:
        return self.left <= t < self.right

    def map_time[U: Time](self, f: Callable[[T], U]) -> "Interval[U]":
        """
        Apply f to both ends of the interval.

        f must be non-decreasing, otherwise the ends would be swapped
        and the interval would be closed on the right.
        This raises a ValueError instead.
        """
        return Interval(f(self.left), f(self.right))

    # Arithmetic with scalars.
    # Interval operands are rejected explicitly, since T is duck-typed:
    # otherwise iv1 + iv2 would compute iv1.left + iv2 = Interval(...) via __radd__.
    # For the same reason, mypy reports the reflected methods as unsafely overlapping
    # (Interval itself would fit the Time protocol), which these checks rule out.

    # shifting by a time value

    def __add__(self, t: T) -> "Interval[T]":
        if isinstance(t, Interval):
            return NotImplemented
        return self.map_time(lambda x: x + t)

    def __radd__(self, t: T) -> "Interval[T]":  # type: ignore[misc]
        if isinstance(t, Interval):
            return NotImplemented
        return self.map_time(lambda x: t + x)

    def __sub__(self, t: T) -> "Interval[T]":
        if isinstance(t, Interval):
            return NotImplemented
        return self.map_time(lambda x: x - t)

    # scaling by a factor (must not be negative, see map_time)
    # (no division: multiply by the reciprocal instead, e.g. iv * Fraction(1, 2))

    def __mul__(self, factor: T) -> "Interval[T]":
        if isinstance(factor, Interval):
            return NotImplemented
        return self.map_time(lambda x: x * factor)

    def __rmul__(self, factor: T) -> "Interval[T]":  # type: ignore[misc]
        if isinstance(factor, Interval):
            return NotImplemented
        return self.map_time(lambda x: factor * x)
