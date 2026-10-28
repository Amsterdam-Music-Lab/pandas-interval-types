import pytest
from fractions import Fraction as F
from pandas_interval_types import Interval, IntervalArray
import pandas_interval_types.interval_relations as ir
import pandas as pd

TIME_TYPES = [
    pytest.param(int, id="int"),
    pytest.param(float, id="float"),
    pytest.param(F, id="Fraction"),
]


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
    # for fn_rel, name_rel in zip(RELATION_FNS, RELATIONS):
    # test array vs scalar
    assert (fn_rel(a_ivs, b_iv) == (RELATIONS == name_rel)).all()

    # test array vs array
    assert (fn_rel(a_ivs, b_iv) == (RELATIONS == name_rel)).all()

    for rel_a, a in zip(RELATIONS, a_ivs):
        # test scalar vs scalar
        assert fn_rel(a, b_iv) == (rel_a == name_rel)

        # test scalar vs array
        assert (fn_rel(a, b_ivs) == (rel_a == name_rel)).all()
