import pytest
from fractions import Fraction as F
from pandas_interval_types import Interval, IntervalArray
import pandas_interval_types.interval_relations as ir
import pandas as pd
import numpy as np

TIME_TYPES = [
    pytest.param(int, id="int"),
    pytest.param(float, id="float"),
    pytest.param(F, id="Fraction"),
]

# Test Allen relations
# ====================


def intervals_a(t):
    return IntervalArray._from_sequence(
        [
            Interval(t(0), t(1)),
            Interval(t(6), t(7)),
            Interval(t(1), t(2)),
            Interval(t(5), t(6)),
            Interval(t(1), t(3)),
            Interval(t(4), t(6)),
            Interval(t(2), t(3)),
            Interval(t(2), t(6)),
            Interval(t(1), t(6)),
            Interval(t(3), t(4)),
            Interval(t(4), t(5)),
            Interval(t(1), t(5)),
            Interval(t(2), t(5)),
        ]
    )


def interval_b(t):
    return Interval(t(2), t(5))


def intervals_b(t):
    return IntervalArray._from_sequence([interval_b(t)] * 13)


RELATIONS = pd.array(
    [
        "precedes",
        "follows",
        "meets",
        "is_met",
        "overlaps",
        "is_overlapped",
        "starts",
        "is_started",
        "contains_strict",
        "during_strict",
        "finishes",
        "is_finished",
        "equal",
    ]
)

RELATION_FNS = pd.array(
    [
        ir.precedes,
        ir.follows,
        ir.meets,
        ir.is_met,
        ir.overlaps,
        ir.is_overlapped,
        ir.starts,
        ir.is_started,
        ir.contains_strict,
        ir.during_strict,
        ir.finishes,
        ir.is_finished,
        ir.equal,
    ]
)

RELATION_TYPES = [
    pytest.param((fn, name), id=name) for fn, name in zip(RELATION_FNS, RELATIONS)
]


@pytest.mark.parametrize("t", TIME_TYPES)
@pytest.mark.parametrize("r", RELATION_TYPES)
def test_allen_relations(t, r):
    fn_rel, name_rel = r
    a_ivs = intervals_a(t)
    b_iv = interval_b(t)
    b_ivs = intervals_b(t)

    # test array vs scalar
    assert (fn_rel(a_ivs, b_iv) == (RELATIONS == name_rel)).all()

    # test array vs array
    assert (fn_rel(a_ivs, b_iv) == (RELATIONS == name_rel)).all()

    for rel_a, a in zip(RELATIONS, a_ivs):
        # test scalar vs scalar
        assert fn_rel(a, b_iv) == (rel_a == name_rel)

        # test scalar vs array
        assert (fn_rel(a, b_ivs) == (rel_a == name_rel)).all()


@pytest.mark.parametrize("t", TIME_TYPES)
def test_allen_exclusive(t):
    rng = np.random.default_rng(4)
    rand_ints = rng.integers(-5, 5, (4, 1000))
    for i in range(200):
        a1, a2, b1, b2 = list(map(t, rand_ints[:, i]))
        a = Interval(min(a1, a2), max(a1, a2))
        b = Interval(min(b1, b2), max(b1, b2))
        rels = RELATION_FNS.map(lambda rel: rel(a, b))
        if rels.sum() != 1:
            print(a, b)
            print(RELATION_FNS.map(lambda rel: rel(a, b)).sum())
            print(RELATIONS[rels])
        assert RELATION_FNS.map(lambda rel: rel(a, b)).sum() == 1


# Test extended interval relations
# ================================

EXTENDED_RELATIONS = [
    (
        ir.intersects,
        [
            "overlaps",
            "is_overlapped",
            "starts",
            "is_started",
            "contains_strict",
            "during_strict",
            "finishes",
            "is_finished",
            "equal",
        ],
    ),
    (ir.contains, ["contains_strict", "is_started", "is_finished", "equal"]),
    (ir.during, ["during_strict", "starts", "finishes", "equal"]),
    (
        ir.contains_start,
        ["overlaps", "starts", "is_started", "is_finished", "equal", "contains_strict"],
    ),
    (
        ir.starts_in,
        ["is_overlapped", "starts", "is_started", "finishes", "equal", "during_strict"],
    ),
    (
        ir.contains_end,
        [
            "is_overlapped",
            "is_started",
            "finishes",
            "is_finished",
            "equal",
            "contains_strict",
        ],
    ),
    (
        ir.ends_in,
        ["overlaps", "starts", "finishes", "is_finished", "equal", "during_strict"],
    ),
    (ir.starts_with, ["starts", "is_started", "equal"]),
    (ir.ends_with, ["finishes", "is_finished", "equal"]),
]


@pytest.mark.parametrize("t", TIME_TYPES)
@pytest.mark.parametrize("r", EXTENDED_RELATIONS)
def test_extended_iv_relations(t, r):
    fn_rel, names_rel = r
    a_ivs = intervals_a(t)
    b_iv = interval_b(t)
    b_ivs = intervals_b(t)

    # test array vs scalar
    try:
        assert (fn_rel(a_ivs, b_iv) == (RELATIONS.isin(names_rel))).all()
    except:
        print(fn_rel)
        print(RELATIONS[fn_rel(a_ivs, b_iv)])

    # test array vs array
    assert (fn_rel(a_ivs, b_iv) == (RELATIONS.isin(names_rel))).all()

    for rel_a, a in zip(RELATIONS, a_ivs):
        # test scalar vs scalar
        assert fn_rel(a, b_iv) == (rel_a in names_rel)

        # test scalar vs array
        assert (fn_rel(a, b_ivs) == (rel_a in names_rel)).all()


# Test empty intervals
# ====================


def empty_cases(t):
    def i(left, right):
        return Interval(t(left), t(right))

    cases = [
        (i(0, 0), i(1, 3), "precedes"),
        (i(1, 1), i(1, 3), "starts"),
        (i(2, 2), i(1, 3), "during_strict"),
        (i(3, 3), i(1, 3), "is_met"),
        (i(4, 4), i(1, 3), "follows"),
        (i(1, 3), i(0, 0), "follows"),
        (i(1, 3), i(1, 1), "is_started"),
        (i(1, 3), i(2, 2), "contains_strict"),
        (i(1, 3), i(3, 3), "meets"),
        (i(1, 3), i(4, 4), "precedes"),
        (i(1, 1), i(2, 2), "precedes"),
        (i(2, 2), i(2, 2), "equal"),
        (i(3, 3), i(2, 2), "follows"),
    ]
    a = IntervalArray._from_sequence([x[0] for x in cases])
    b = IntervalArray._from_sequence([x[1] for x in cases])
    expected = pd.array([x[2] for x in cases])
    return a, b, expected


@pytest.mark.parametrize("t", TIME_TYPES)
@pytest.mark.parametrize("r", RELATION_TYPES)
def test_allen_empty(t, r):
    fn_rel, name_rel = r
    a_ivs, b_ivs, expected = empty_cases(t)

    # test scalars
    for a, b, exp in zip(a_ivs, b_ivs, expected):
        assert fn_rel(a, b) == (name_rel == exp)

    # test array vs scalar and scalar vs array
    for i, (a, b, exp) in enumerate(zip(a_ivs, b_ivs, expected)):
        assert (fn_rel(a_ivs[[i]], b) == (name_rel == exp)).all()
        assert (fn_rel(a, b_ivs[[i]]) == (name_rel == exp)).all()

    # test array vs array
    assert (fn_rel(a_ivs, b_ivs) == (expected == name_rel)).all()


@pytest.mark.parametrize("t", TIME_TYPES)
@pytest.mark.parametrize("r", EXTENDED_RELATIONS)
def test_extended_empty(t, r):
    fn_rel, names_rel = r
    a_ivs, b_ivs, expected = empty_cases(t)

    # test scalars
    for a, b, exp in zip(a_ivs, b_ivs, expected):
        assert fn_rel(a, b) == (exp in names_rel)

    # test array vs scalar and scalar vs array
    for i, (a, b, exp) in enumerate(zip(a_ivs, b_ivs, expected)):
        assert (fn_rel(a_ivs[[i]], b) == (exp in names_rel)).all()
        assert (fn_rel(a, b_ivs[[i]]) == (exp in names_rel)).all()

    # test array vs array
    assert (fn_rel(a_ivs, b_ivs) == (expected.isin(names_rel))).all()


# Point Relations
# ===============


@pytest.mark.parametrize("t", TIME_TYPES)
def test_point_relations(t):
    def i(left, right):
        return Interval(t(left), t(right))

    assert ir.point_before_interval(t(0), i(1, 3))
    assert not ir.point_before_interval(t(1), i(1, 3))
    assert not ir.point_before_interval(t(2), i(1, 3))
    assert not ir.point_before_interval(t(3), i(1, 3))
    assert not ir.point_before_interval(t(4), i(1, 3))

    assert not ir.point_starts_interval(t(0), i(1, 3))
    assert ir.point_starts_interval(t(1), i(1, 3))
    assert not ir.point_starts_interval(t(2), i(1, 3))
    assert not ir.point_starts_interval(t(3), i(1, 3))
    assert not ir.point_starts_interval(t(4), i(1, 3))

    assert not ir.point_strictly_inside_interval(t(0), i(1, 3))
    assert not ir.point_strictly_inside_interval(t(1), i(1, 3))
    assert ir.point_strictly_inside_interval(t(2), i(1, 3))
    assert not ir.point_strictly_inside_interval(t(3), i(1, 3))
    assert not ir.point_strictly_inside_interval(t(4), i(1, 3))

    assert not ir.point_ends_interval(t(0), i(1, 3))
    assert not ir.point_ends_interval(t(1), i(1, 3))
    assert not ir.point_ends_interval(t(2), i(1, 3))
    assert ir.point_ends_interval(t(3), i(1, 3))
    assert not ir.point_ends_interval(t(4), i(1, 3))

    assert not ir.point_after_interval(t(0), i(1, 3))
    assert not ir.point_after_interval(t(1), i(1, 3))
    assert not ir.point_after_interval(t(2), i(1, 3))
    assert not ir.point_after_interval(t(3), i(1, 3))
    assert ir.point_after_interval(t(4), i(1, 3))

    assert not ir.point_in_interval(t(0), i(1, 3))
    assert ir.point_in_interval(t(1), i(1, 3))
    assert ir.point_in_interval(t(2), i(1, 3))
    assert not ir.point_in_interval(t(3), i(1, 3))
    assert not ir.point_in_interval(t(4), i(1, 3))


@pytest.mark.parametrize("t", TIME_TYPES)
def test_point_empty_relations(t):
    def i(left, right):
        return Interval(t(left), t(right))

    assert ir.point_before_interval(t(0), i(1, 1))
    assert not ir.point_before_interval(t(1), i(1, 1))
    assert not ir.point_before_interval(t(2), i(1, 1))

    assert not ir.point_starts_interval(t(0), i(1, 1))
    assert ir.point_starts_interval(t(1), i(1, 1))
    assert not ir.point_starts_interval(t(2), i(1, 1))

    assert not ir.point_strictly_inside_interval(t(0), i(1, 1))
    assert not ir.point_strictly_inside_interval(t(1), i(1, 1))
    assert not ir.point_strictly_inside_interval(t(2), i(1, 1))

    assert not ir.point_ends_interval(t(0), i(1, 1))
    assert ir.point_ends_interval(t(1), i(1, 1))
    assert not ir.point_ends_interval(t(2), i(1, 1))

    assert not ir.point_after_interval(t(0), i(1, 1))
    assert not ir.point_after_interval(t(1), i(1, 1))
    assert ir.point_after_interval(t(2), i(1, 1))

    assert not ir.point_in_interval(t(0), i(1, 1))
    assert ir.point_in_interval(t(1), i(1, 1))
    assert not ir.point_in_interval(t(2), i(1, 1))
