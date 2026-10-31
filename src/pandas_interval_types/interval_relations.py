from typing import Literal, overload
from pandas_interval_types.interval import Interval, Time
from pandas_interval_types.array import IntervalArray
import numpy.typing as npt
import numpy as np

# TODO: use overloads for more fine-grained types in relations

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
    return (a.left > b.left) & (a.right == b.right) & (a.left < b.right)


def is_finished(a: IntervalLike, b: IntervalLike) -> Boolish:
    return finishes(b, a)


def equal(a: IntervalLike, b: IntervalLike) -> Boolish:
    return (a.left == b.left) & (a.right == b.right)


# Extra relations
# ---------------


def intersects(a: IntervalLike, b: IntervalLike) -> Boolish:
    a_closed = a.left == a.right
    b_closed = b.left == b.right
    a_after_b = (a.right > b.left) | (a_closed & (a.right >= b.left))
    b_after_a = (b.right > a.left) | (b_closed & (b.right >= a.left))
    return a_after_b & b_after_a


def contains(a: IntervalLike, b: IntervalLike) -> Boolish:
    """constains_strict or is_started or is_finished or equal"""
    b_open = b.left != b.right
    a_closed = a.left == a.right
    left_in = a.left <= b.left
    right_in = (a.right > b.right) | ((b_open | a_closed) & (a.right >= b.right))
    return left_in & right_in


def during(a: IntervalLike, b: IntervalLike) -> Boolish:
    """during_strict or starts or finishes or equal"""
    return contains(b, a)


def contains_start(a: IntervalLike, b: IntervalLike) -> Boolish:
    a_closed = a.left == a.right
    left = a.left <= b.left
    right = (a.right > b.left) | (a_closed & (a.right >= b.left))
    return left & right


def starts_in(a: IntervalLike, b: IntervalLike) -> Boolish:
    return contains_start(b, a)


def contains_end(a: IntervalLike, b: IntervalLike) -> Boolish:
    a_closed = a.left == a.right
    b_closed = b.left == b.right
    a_open = a.left != a.right
    b_open = b.left != b.right
    left = (b_closed & (a.left <= b.right)) | (a.left < b.right)
    right = ((a_closed | b_open) & (a.right >= b.right)) | (a.right > b.right)
    return left & right


def ends_in(a: IntervalLike, b: IntervalLike) -> Boolish:
    return contains_end(b, a)


def starts_with(a: IntervalLike, b: IntervalLike) -> Boolish:
    return a.left == b.left


def ends_with(a: IntervalLike, b: IntervalLike) -> Boolish:
    return (a.right == b.right) & ((a.left == a.right) == (b.left == b.right))


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


# Operations
# ==========


@overload
def intersection(a: Interval, b: Interval) -> Interval | None: ...
@overload
def intersection(a: IntervalLike, b: IntervalLike) -> IntervalLike: ...
def intersection(a: IntervalLike, b: IntervalLike) -> IntervalLike | None:
    """Returns the intersection of intervals a and b."""
    left = np.maximum(a.left, b.left)
    right = np.minimum(a.right, b.right)
    valid = (left < right) | (
        (left == right) & point_in_interval(left, a) & point_in_interval(left, b)
    )
    if isinstance(a, Interval) and isinstance(b, Interval):
        if valid:
            return Interval(left, right)
        else:
            return None
    else:
        left[~valid] = None
        right[~valid] = None
        return IntervalArray.from_arrays(left, right)
