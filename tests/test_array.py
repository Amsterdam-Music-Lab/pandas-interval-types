from fractions import Fraction as F

import numpy as np
import pandas as pd
import pytest
from pandas_fractions import FractionArray, FractionDType

from pandas_intervals import Interval, IntervalArray, IntervalDType


def frac(values):
    return FractionArray._from_sequence(values)


def test_dtype_names():
    assert IntervalDType(FractionDType()).name == "span[frac]"
    assert pd.api.types.pandas_dtype("span[frac]") == IntervalDType(FractionDType())
    assert pd.api.types.pandas_dtype("span") == IntervalDType()
    # pandas' own interval dtype is unaffected
    assert isinstance(pd.api.types.pandas_dtype("interval[int64]"), pd.IntervalDtype)


def test_dtype_canonical_subtype():
    # numpy subtypes are replaced by extension dtypes, which can hold NA
    assert IntervalDType("int64").subtype == pd.Int64Dtype()
    assert IntervalDType("float64").subtype == pd.Float64Dtype()
    assert pd.api.types.pandas_dtype("span[int64]") == IntervalDType("Int64")


@pytest.mark.parametrize(
    "ivs, subtype",
    [
        ([Interval(F(0), F(1, 2))], FractionDType()),
        ([Interval(0, F(1, 2))], FractionDType()),  # int and Fraction stay exact
        ([Interval(0, 1)], pd.Int64Dtype()),
        ([Interval(0, 0.5)], pd.Float64Dtype()),
        ([Interval(F(0), 0.5)], np.dtype(object)),  # no implicit float conversion
    ],
)
def test_infer_subtype(ivs, subtype):
    arr = pd.array(ivs + [None], dtype="span")
    assert arr.dtype == IntervalDType(subtype)
    assert arr[0] == ivs[0]
    assert pd.isna(arr[1])


def test_from_arrays():
    arr = IntervalArray.from_arrays(frac([0, F(1, 2), None]), [F(1, 2), 1, None])
    assert arr.dtype == IntervalDType(FractionDType())
    assert list(arr[:2]) == [Interval(0, F(1, 2)), Interval(F(1, 2), 1)]
    assert arr.isna().tolist() == [False, False, True]
    assert isinstance(arr.left, FractionArray) and isinstance(arr.right, FractionArray)
    assert arr.duration.tolist()[:2] == [F(1, 2), F(1, 2)]
    # the subtype can be given explicitly
    as_float = IntervalArray.from_arrays([0, 1], [1, 2], dtype="span[float64]")
    assert as_float.dtype == IntervalDType("Float64")


def test_from_arrays_invalid():
    with pytest.raises(ValueError, match="length"):
        IntervalArray.from_arrays([0, 1], [1])
    with pytest.raises(ValueError, match="missing"):
        IntervalArray.from_arrays([0, None], [1, 2])
    with pytest.raises(ValueError, match="before"):
        IntervalArray.from_arrays([0, 2], [1, 1])
    with pytest.raises(TypeError):
        pd.array([(0, 1)], dtype="span")


def test_setitem():
    arr = pd.array([Interval(F(0), F(1)), Interval(F(1), F(2))], dtype="span")
    arr[0] = Interval(F(1, 2), F(1))
    arr[1] = None
    assert arr[0] == Interval(F(1, 2), F(1))
    assert pd.isna(arr[1])
    arr[:] = [Interval(0, 1), Interval(1, 2)]
    assert list(arr) == [Interval(0, 1), Interval(1, 2)]
    with pytest.raises(TypeError):
        arr[0] = (0, 1)


def test_astype():
    arr = pd.array([Interval(F(1, 2), F(3, 2)), None], dtype="span")
    as_float = arr.astype("span[float64]")
    assert as_float.dtype == IntervalDType("Float64")
    assert as_float[0] == Interval(0.5, 1.5)
    assert pd.isna(as_float[1])
    objects = arr.astype(object)
    assert objects.dtype == object and objects[0] == arr[0]


def test_concat_common_subtype():
    fractions = pd.Series(pd.array([Interval(F(1, 2), F(1))], dtype="span"))
    ints = pd.Series(pd.array([Interval(0, 1)], dtype="span"))
    result = pd.concat([fractions, ints], ignore_index=True)
    assert result.dtype == IntervalDType(FractionDType())
    assert list(result) == [Interval(F(1, 2), F(1)), Interval(0, 1)]


def test_compare():
    arr = pd.array(
        [Interval(F(0), F(1)), Interval(F(0), F(2)), Interval(F(1), F(1)), None],
        dtype="span",
    )
    iv = Interval(0, 2)
    assert (arr == iv).tolist() == [False, True, False, False]
    assert (arr != iv).tolist() == [True, False, True, True]
    assert (arr < iv).tolist() == [True, False, False, False]
    assert (arr <= iv).tolist() == [True, True, False, False]
    assert (arr > iv).tolist() == [False, False, True, False]
    assert (arr == "a").tolist() == [False] * 4
    with pytest.raises(TypeError):
        arr < "a"


def test_sort_and_group():
    ivs = [Interval(F(1), F(2)), Interval(F(0), F(2)), None, Interval(F(0), F(1))]
    ser = pd.Series(pd.array(ivs, dtype="span"))
    assert list(ser.sort_values().iloc[:3]) == sorted(ivs[:2] + ivs[3:])
    assert pd.isna(ser.sort_values().iloc[3])
    counts = pd.Series(pd.array(ivs[:2] * 2, dtype="span")).value_counts()
    assert counts.to_dict() == {ivs[0]: 2, ivs[1]: 2}
