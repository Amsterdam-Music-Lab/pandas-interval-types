"""
Run pandas' test suite for extension arrays against IntervalArray.

The suite checks that IntervalArray implements the ExtensionArray interface
consistently with pandas' expectations. It is configured using the fixtures and
the overridden methods of TestIntervalArray below.
The intervals have fraction ends, the main use case.
"""

from fractions import Fraction as F

import pandas as pd
import pytest
from pandas.tests.extension import base
from pandas_fractions import FractionDType

from pandas_interval_types import Interval, IntervalArray, IntervalDType


def iv(left, right):
    return Interval(F(left), F(right))


def intervals(values):
    return IntervalArray._from_sequence(values, dtype=IntervalDType(FractionDType()))


@pytest.fixture
def dtype():
    return IntervalDType(FractionDType())


@pytest.fixture
def data():
    return intervals([iv(F(k, 2), F(k, 2) + F(k % 3 + 1, 4)) for k in range(10)])


@pytest.fixture
def data_missing():
    return intervals([None, iv(0, F(1, 2))])


@pytest.fixture
def data_for_sorting():
    # sorted by left first, then by right: c < a < b
    return intervals([iv(0, 2), iv(1, 2), iv(0, 1)])


@pytest.fixture
def data_missing_for_sorting():
    return intervals([iv(0, 2), None, iv(0, 1)])


@pytest.fixture
def data_for_grouping():
    a, b, c = iv(0, 1), iv(0, 2), iv(1, 2)
    return intervals([b, b, None, None, a, a, b, c])


@pytest.fixture
def na_cmp():
    return lambda x, y: pd.isna(x) and pd.isna(y)


class TestIntervalArray(base.ExtensionTests):
    def _supports_reduction(self, ser, op_name):
        return op_name in ("count", "min", "max")

    # unsupported by design

    @pytest.mark.xfail(reason="pandas reads intervals from JSON as dicts")
    def test_json_roundtrip(self, data):
        super().test_json_roundtrip(data)

    @pytest.mark.xfail(
        reason="pandas turns a list of dataclasses (like Interval) into rows of fields"
    )
    def test_from_dtype(self, data):
        super().test_from_dtype(data)
