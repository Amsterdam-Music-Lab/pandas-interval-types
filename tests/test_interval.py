from dataclasses import FrozenInstanceError
from fractions import Fraction as F

import pytest

from pandas_intervals import Interval

# the same interval [1/2, 3/2) with different time types, where representable
TIME_TYPES = [
    pytest.param(int, id="int"),
    pytest.param(float, id="float"),
    pytest.param(F, id="Fraction"),
]


@pytest.mark.parametrize("t", TIME_TYPES)
def test_construction(t):
    iv = Interval(t(1), t(3))
    assert iv.left == t(1) and iv.right == t(3)
    assert type(iv.left) is t
    assert Interval.from_duration(t(1), t(2)) == iv
    assert iv.duration == t(2)
    # zero-length intervals are allowed
    assert Interval(t(1), t(1)).duration == 0


@pytest.mark.parametrize("t", TIME_TYPES)
def test_invalid(t):
    with pytest.raises(ValueError, match="before"):
        Interval(t(2), t(1))
    with pytest.raises(ValueError):
        Interval.from_duration(t(2), t(-1))


def test_immutable():
    iv = Interval(0, 1)
    with pytest.raises(FrozenInstanceError):
        iv.left = 2


def test_contains():
    iv = Interval(F(1, 2), F(3, 2))
    assert F(1, 2) in iv  # closed on the left
    assert 1 in iv
    assert F(3, 2) not in iv  # open on the right
    assert 0 not in iv
    assert 0 not in Interval(0, 0)  # zero-length intervals contain nothing


def test_equality_and_hash_across_types():
    assert Interval(1, 2) == Interval(1.0, F(2))
    assert hash(Interval(1, 2)) == hash(Interval(1.0, F(2)))
    assert Interval(1, 2) != Interval(1, 3)
    assert len({Interval(1, 2), Interval(F(1), 2.0), Interval(1, 3)}) == 2


def test_order():
    ivs = [Interval(1, 3), Interval(0, 5), Interval(1, 2)]
    assert sorted(ivs) == [Interval(0, 5), Interval(1, 2), Interval(1, 3)]


def test_map_time():
    iv = Interval(F(1, 2), F(3, 2))
    as_float = iv.map_time(float)
    assert as_float == Interval(0.5, 1.5)
    assert type(as_float.left) is float
    assert iv.map_time(lambda x: x**2) == Interval(F(1, 4), F(9, 4))
    # decreasing functions would swap the ends
    with pytest.raises(ValueError):
        iv.map_time(lambda x: -x)


@pytest.mark.parametrize("t", TIME_TYPES)
def test_shift(t):
    iv = Interval(t(1), t(3))
    assert iv + t(2) == Interval(t(3), t(5))
    assert t(2) + iv == Interval(t(3), t(5))
    assert iv - t(1) == Interval(t(0), t(2))
    assert (iv + t(2)).duration == iv.duration
    with pytest.raises(TypeError):
        t(1) - iv  # would mirror the interval


@pytest.mark.parametrize("t", TIME_TYPES)
def test_scale(t):
    iv = Interval(t(1), t(3))
    assert iv * t(2) == Interval(t(2), t(6))
    assert t(2) * iv == Interval(t(2), t(6))
    assert (iv * t(2)).duration == 2 * iv.duration
    assert iv * t(0) == Interval(t(0), t(0))
    with pytest.raises(ValueError):
        iv * t(-1)


def test_no_division():
    iv = Interval(F(1, 2), F(3, 2))
    with pytest.raises(TypeError):
        iv / 2
    # multiply by the reciprocal instead
    assert iv * F(1, 2) == Interval(F(1, 4), F(3, 4))


def test_mixed_types():
    # int and Fraction stay exact, Fraction and float become float
    assert type((Interval(0, F(1, 2)) + 1).right) is F
    assert type((Interval(F(0), F(1, 2)) * 0.5).right) is float


def test_no_interval_or_collection_operands():
    # operations between intervals are separate functions
    with pytest.raises(TypeError):
        Interval(0, 1) + Interval(1, 2)
    with pytest.raises(TypeError):
        Interval(0, 1) * Interval(1, 2)
    with pytest.raises(TypeError):
        Interval(0, 1) - Interval(1, 2)
    with pytest.raises(TypeError):
        1 - Interval(1, 2)
    # collections (e.g. arrays) are not time values
    with pytest.raises(TypeError):
        Interval(0, 1) + [1]
