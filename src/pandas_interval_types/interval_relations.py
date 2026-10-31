import typing
from typing import Literal, overload, Callable
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


@overload
def precedes(a: Interval, b: Interval) -> bool: ...
@overload
def precedes(a: Interval, b: IntervalArray) -> npt.NDArray[np.bool]: ...
@overload
def precedes(a: IntervalArray, b: Interval) -> npt.NDArray[np.bool]: ...
@overload
def precedes(a: IntervalArray, b: IntervalArray) -> npt.NDArray[np.bool]: ...
def precedes(a, b):
    return a.right < b.left


@overload
def follows(a: Interval, b: Interval) -> bool: ...
@overload
def follows(a: Interval, b: IntervalArray) -> npt.NDArray[np.bool]: ...
@overload
def follows(a: IntervalArray, b: Interval) -> npt.NDArray[np.bool]: ...
@overload
def follows(a: IntervalArray, b: IntervalArray) -> npt.NDArray[np.bool]: ...
def follows(a, b):
    return precedes(b, a)


@overload
def meets(a: Interval, b: Interval) -> bool: ...
@overload
def meets(a: Interval, b: IntervalArray) -> npt.NDArray[np.bool]: ...
@overload
def meets(a: IntervalArray, b: Interval) -> npt.NDArray[np.bool]: ...
@overload
def meets(a: IntervalArray, b: IntervalArray) -> npt.NDArray[np.bool]: ...
def meets(a, b):
    return (a.right == b.left) & (a.left < b.left)


@overload
def is_met(a: Interval, b: Interval) -> bool: ...
@overload
def is_met(a: Interval, b: IntervalArray) -> npt.NDArray[np.bool]: ...
@overload
def is_met(a: IntervalArray, b: Interval) -> npt.NDArray[np.bool]: ...
@overload
def is_met(a: IntervalArray, b: IntervalArray) -> npt.NDArray[np.bool]: ...
def is_met(a, b):
    return meets(b, a)


@overload
def overlaps(a: Interval, b: Interval) -> bool: ...
@overload
def overlaps(a: Interval, b: IntervalArray) -> npt.NDArray[np.bool]: ...
@overload
def overlaps(a: IntervalArray, b: Interval) -> npt.NDArray[np.bool]: ...
@overload
def overlaps(a: IntervalArray, b: IntervalArray) -> npt.NDArray[np.bool]: ...
def overlaps(a, b):
    return (a.left < b.left) & (a.right > b.left) & (a.right < b.right)


@overload
def is_overlapped(a: Interval, b: Interval) -> bool: ...
@overload
def is_overlapped(a: Interval, b: IntervalArray) -> npt.NDArray[np.bool]: ...
@overload
def is_overlapped(a: IntervalArray, b: Interval) -> npt.NDArray[np.bool]: ...
@overload
def is_overlapped(a: IntervalArray, b: IntervalArray) -> npt.NDArray[np.bool]: ...
def is_overlapped(a, b):
    return overlaps(b, a)


@overload
def starts(a: Interval, b: Interval) -> bool: ...
@overload
def starts(a: Interval, b: IntervalArray) -> npt.NDArray[np.bool]: ...
@overload
def starts(a: IntervalArray, b: Interval) -> npt.NDArray[np.bool]: ...
@overload
def starts(a: IntervalArray, b: IntervalArray) -> npt.NDArray[np.bool]: ...
def starts(a, b):
    return (a.left == b.left) & (a.right < b.right)


@overload
def is_started(a: Interval, b: Interval) -> bool: ...
@overload
def is_started(a: Interval, b: IntervalArray) -> npt.NDArray[np.bool]: ...
@overload
def is_started(a: IntervalArray, b: Interval) -> npt.NDArray[np.bool]: ...
@overload
def is_started(a: IntervalArray, b: IntervalArray) -> npt.NDArray[np.bool]: ...
def is_started(a, b):
    return starts(b, a)


@overload
def contains_strict(a: Interval, b: Interval) -> bool: ...
@overload
def contains_strict(a: Interval, b: IntervalArray) -> npt.NDArray[np.bool]: ...
@overload
def contains_strict(a: IntervalArray, b: Interval) -> npt.NDArray[np.bool]: ...
@overload
def contains_strict(a: IntervalArray, b: IntervalArray) -> npt.NDArray[np.bool]: ...
def contains_strict(a, b):
    return (a.left < b.left) & (a.right > b.right)


@overload
def during_strict(a: Interval, b: Interval) -> bool: ...
@overload
def during_strict(a: Interval, b: IntervalArray) -> npt.NDArray[np.bool]: ...
@overload
def during_strict(a: IntervalArray, b: Interval) -> npt.NDArray[np.bool]: ...
@overload
def during_strict(a: IntervalArray, b: IntervalArray) -> npt.NDArray[np.bool]: ...
def during_strict(a, b):
    return contains_strict(b, a)


@overload
def finishes(a: Interval, b: Interval) -> bool: ...
@overload
def finishes(a: Interval, b: IntervalArray) -> npt.NDArray[np.bool]: ...
@overload
def finishes(a: IntervalArray, b: Interval) -> npt.NDArray[np.bool]: ...
@overload
def finishes(a: IntervalArray, b: IntervalArray) -> npt.NDArray[np.bool]: ...
def finishes(a, b):
    return (a.left > b.left) & (a.right == b.right) & (a.left < b.right)


@overload
def is_finished(a: Interval, b: Interval) -> bool: ...
@overload
def is_finished(a: Interval, b: IntervalArray) -> npt.NDArray[np.bool]: ...
@overload
def is_finished(a: IntervalArray, b: Interval) -> npt.NDArray[np.bool]: ...
@overload
def is_finished(a: IntervalArray, b: IntervalArray) -> npt.NDArray[np.bool]: ...
def is_finished(a, b):
    return finishes(b, a)


@overload
def equal(a: Interval, b: Interval) -> bool: ...
@overload
def equal(a: Interval, b: IntervalArray) -> npt.NDArray[np.bool]: ...
@overload
def equal(a: IntervalArray, b: Interval) -> npt.NDArray[np.bool]: ...
@overload
def equal(a: IntervalArray, b: IntervalArray) -> npt.NDArray[np.bool]: ...
def equal(a, b):
    return (a.left == b.left) & (a.right == b.right)


# Extra relations
# ---------------


@overload
def intersects(a: Interval, b: Interval) -> bool: ...
@overload
def intersects(a: Interval, b: IntervalArray) -> npt.NDArray[np.bool]: ...
@overload
def intersects(a: IntervalArray, b: Interval) -> npt.NDArray[np.bool]: ...
@overload
def intersects(a: IntervalArray, b: IntervalArray) -> npt.NDArray[np.bool]: ...
def intersects(a, b):
    a_closed = a.left == a.right
    b_closed = b.left == b.right
    a_after_b = (a.right > b.left) | (a_closed & (a.right >= b.left))
    b_after_a = (b.right > a.left) | (b_closed & (b.right >= a.left))
    return a_after_b & b_after_a


@overload
def contains(a: Interval, b: Interval) -> bool: ...
@overload
def contains(a: Interval, b: IntervalArray) -> npt.NDArray[np.bool]: ...
@overload
def contains(a: IntervalArray, b: Interval) -> npt.NDArray[np.bool]: ...
@overload
def contains(a: IntervalArray, b: IntervalArray) -> npt.NDArray[np.bool]: ...
def contains(a, b):
    """constains_strict or is_started or is_finished or equal"""
    b_open = b.left != b.right
    a_closed = a.left == a.right
    left_in = a.left <= b.left
    right_in = (a.right > b.right) | ((b_open | a_closed) & (a.right >= b.right))
    return left_in & right_in


@overload
def during(a: Interval, b: Interval) -> bool: ...
@overload
def during(a: Interval, b: IntervalArray) -> npt.NDArray[np.bool]: ...
@overload
def during(a: IntervalArray, b: Interval) -> npt.NDArray[np.bool]: ...
@overload
def during(a: IntervalArray, b: IntervalArray) -> npt.NDArray[np.bool]: ...
def during(a, b):
    """during_strict or starts or finishes or equal"""
    return contains(b, a)


@overload
def contains_start(a: Interval, b: Interval) -> bool: ...
@overload
def contains_start(a: Interval, b: IntervalArray) -> npt.NDArray[np.bool]: ...
@overload
def contains_start(a: IntervalArray, b: Interval) -> npt.NDArray[np.bool]: ...
@overload
def contains_start(a: IntervalArray, b: IntervalArray) -> npt.NDArray[np.bool]: ...
def contains_start(a, b):
    a_closed = a.left == a.right
    left = a.left <= b.left
    right = (a.right > b.left) | (a_closed & (a.right >= b.left))
    return left & right


@overload
def starts_in(a: Interval, b: Interval) -> bool: ...
@overload
def starts_in(a: Interval, b: IntervalArray) -> npt.NDArray[np.bool]: ...
@overload
def starts_in(a: IntervalArray, b: Interval) -> npt.NDArray[np.bool]: ...
@overload
def starts_in(a: IntervalArray, b: IntervalArray) -> npt.NDArray[np.bool]: ...
def starts_in(a, b):
    return contains_start(b, a)


@overload
def contains_end(a: Interval, b: Interval) -> bool: ...
@overload
def contains_end(a: Interval, b: IntervalArray) -> npt.NDArray[np.bool]: ...
@overload
def contains_end(a: IntervalArray, b: Interval) -> npt.NDArray[np.bool]: ...
@overload
def contains_end(a: IntervalArray, b: IntervalArray) -> npt.NDArray[np.bool]: ...
def contains_end(a, b):
    a_closed = a.left == a.right
    b_closed = b.left == b.right
    # a_open = a.left != a.right
    b_open = b.left != b.right
    left = (b_closed & (a.left <= b.right)) | (a.left < b.right)
    right = ((a_closed | b_open) & (a.right >= b.right)) | (a.right > b.right)
    return left & right


@overload
def ends_in(a: Interval, b: Interval) -> bool: ...
@overload
def ends_in(a: Interval, b: IntervalArray) -> npt.NDArray[np.bool]: ...
@overload
def ends_in(a: IntervalArray, b: Interval) -> npt.NDArray[np.bool]: ...
@overload
def ends_in(a: IntervalArray, b: IntervalArray) -> npt.NDArray[np.bool]: ...
def ends_in(a, b):
    return contains_end(b, a)


@overload
def starts_with(a: Interval, b: Interval) -> bool: ...
@overload
def starts_with(a: Interval, b: IntervalArray) -> npt.NDArray[np.bool]: ...
@overload
def starts_with(a: IntervalArray, b: Interval) -> npt.NDArray[np.bool]: ...
@overload
def starts_with(a: IntervalArray, b: IntervalArray) -> npt.NDArray[np.bool]: ...
def starts_with(a, b):
    return a.left == b.left


@overload
def ends_with(a: Interval, b: Interval) -> bool: ...
@overload
def ends_with(a: Interval, b: IntervalArray) -> npt.NDArray[np.bool]: ...
@overload
def ends_with(a: IntervalArray, b: Interval) -> npt.NDArray[np.bool]: ...
@overload
def ends_with(a: IntervalArray, b: IntervalArray) -> npt.NDArray[np.bool]: ...
def ends_with(a, b):
    return (a.right == b.right) & ((a.left == a.right) == (b.left == b.right))


# Point-Interval Relations
# ========================

# Relation Functions
# ------------------


@overload
def point_before_interval[T: Time](a: T, i: Interval[T]) -> bool: ...
@overload
def point_before_interval(a, i: Interval | IntervalArray) -> npt.NDArray[np.bool]: ...
def point_before_interval(a, i):
    return a < i.left


@overload
def point_starts_interval[T: Time](a: T, i: Interval[T]) -> bool: ...
@overload
def point_starts_interval(a, i: Interval | IntervalArray) -> npt.NDArray[np.bool]: ...
def point_starts_interval(a, i):
    return a == i.left


@overload
def point_strictly_inside_interval[T: Time](a: T, i: Interval[T]) -> bool: ...
@overload
def point_strictly_inside_interval(
    a, i: Interval | IntervalArray
) -> npt.NDArray[np.bool]: ...
def point_strictly_inside_interval(a, i) -> Boolish:
    """Returns True if the point is strictly within the interval."""
    return (a > i.left) & (a < i.right)


@overload
def point_ends_interval[T: Time](a: T, i: Interval[T]) -> bool: ...
@overload
def point_ends_interval(a, i: Interval | IntervalArray) -> npt.NDArray[np.bool]: ...
def point_ends_interval(a, i):
    return a == i.right


@overload
def point_after_interval[T: Time](a: T, i: Interval[T]) -> bool: ...
@overload
def point_after_interval(a, i: Interval | IntervalArray) -> npt.NDArray[np.bool]: ...
def point_after_interval(a, i):
    return a > i.right


@overload
def point_in_interval[T: Time](a: T, i: Interval[T]) -> bool: ...
@overload
def point_in_interval(a, i: Interval | IntervalArray) -> npt.NDArray[np.bool]: ...
def point_in_interval(a, i):
    return ((a >= i.left) & (a < i.right)) | ((a == i.left) & (a == i.right))


# Operations
# ==========


@overload
def intersection(a: Interval, b: Interval) -> Interval | None: ...
@overload
def intersection(a: IntervalLike, b: IntervalLike) -> IntervalLike: ...
def intersection(a, b):
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
