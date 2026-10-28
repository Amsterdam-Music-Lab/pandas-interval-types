from pandas_interval_types.interval import Interval, Time
from pandas_interval_types.array import IntervalArray
import numpy.typing as npt
import numpy as np

IntervalLike = Interval | IntervalArray

Boolish = bool | npt.NDArray[np.bool]

# Interval Relations
# ==================

# Allen relations
# ---------------


def precedes(a: IntervalLike, b: IntervalLike) -> Boolish:
    return a.right < b.left


def follows(a: IntervalLike, b: IntervalLike) -> Boolish:
    return precedes(b, a)


def meets(a: IntervalLike, b: IntervalLike) -> Boolish:
    return (a.right == b.left) & (a.left < b.left)


def is_met(a: IntervalLike, b: IntervalLike) -> Boolish:
    return meets(b, a)


def overlaps(a: IntervalLike, b: IntervalLike) -> Boolish:
    return (a.left < b.left) & (a.right > b.left) & (a.right < b.right)


def is_overlapped(a: IntervalLike, b: IntervalLike) -> Boolish:
    return overlaps(b, a)


def starts(a: IntervalLike, b: IntervalLike) -> Boolish:
    return (a.left == b.left) & (a.right < b.right)


def is_started(a: IntervalLike, b: IntervalLike) -> Boolish:
    return starts(b, a)


def contains_strict(a: IntervalLike, b: IntervalLike) -> Boolish:
    return (a.left < b.left) & (a.right > b.right)


def during_strict(a: IntervalLike, b: IntervalLike) -> Boolish:
    return contains_strict(b, a)


def finishes(a: IntervalLike, b: IntervalLike) -> Boolish:
    return (a.left > b.left) & (a.right == b.right)


def is_finished(a: IntervalLike, b: IntervalLike) -> Boolish:
    return finishes(b, a)


def equal(a: IntervalLike, b: IntervalLike) -> Boolish:
    return (a.left == b.left) & (a.right == b.right)


# Extra relations
# ---------------


def intersects(a: IntervalLike, b: IntervalLike) -> Boolish:
    return (a.right > b.left) & (b.right > a.left)


def contains(a: IntervalLike, b: IntervalLike) -> Boolish:
    """constains_strict or is_started or is_finished"""
    return (a.left <= b.left) & (a.right >= b.right)


def during(a: IntervalLike, b: IntervalLike) -> Boolish:
    """during_strict or starts or finishes"""
    return contains(b, a)


def contains_start(a: IntervalLike, b: IntervalLike) -> Boolish:
    return (a.left <= b.left) & (a.right > b.left)


def starts_in(a: IntervalLike, b: IntervalLike) -> Boolish:
    return contains_start(b, a)


def contains_end(a: IntervalLike, b: IntervalLike) -> Boolish:
    return (a.left < b.right) & (a.right >= b.right)


def ends_in(a: IntervalLike, b: IntervalLike) -> Boolish:
    return contains_end(b, a)


def starts_with(a: IntervalLike, b: IntervalLike) -> Boolish:
    return a.left == b.left


def ends_with(a: IntervalLike, b: IntervalLike) -> Boolish:
    return a.right == b.right


# Point-Interval Relations
# ========================

# Relation Functions
# ------------------


def point_before_interval[T: Time](a: T, i: Interval[T] | IntervalArray) -> Boolish:
    return a < i.left


def point_starts_interval[T: Time](a: T, i: Interval[T] | IntervalArray) -> Boolish:
    return a == i.left


def point_strictly_inside_interval[T: Time](
    a: T, i: Interval[T] | IntervalArray
) -> Boolish:
    """Returns True if the point is strictly within the interval."""
    return (a > i.left) & (a < i.right)


def point_ends_interval[T: Time](a: T, i: Interval[T] | IntervalArray) -> Boolish:
    return a == i.right


def point_after_interval[T: Time](a: T, i: Interval[T] | IntervalArray) -> Boolish:
    return a > i.right


def point_in_interval[T: Time](a: T, i: Interval[T] | IntervalArray) -> Boolish:
    return ((a >= i.left) & (a < i.right)) | ((a == i.left) & (a == i.right))
