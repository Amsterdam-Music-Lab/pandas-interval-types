from decimal import Decimal
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
        ([Interval(Decimal(0), Decimal(1))], np.dtype(object)),  # no own dtype
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


@pytest.mark.parametrize(
    "construct",
    [
        lambda: pd.array([Interval(F(0), 0.5)], dtype="span"),
        lambda: pd.array([Interval(F(0), F(1)), Interval(0.5, 1.5)], dtype="span"),
        lambda: IntervalArray.from_arrays(frac([0]), [0.5]),
        lambda: pd.array([Interval(F(0), F(1))] * 2, dtype="span") + [F(1, 2), 0.5],
    ],
)
def test_lossy_mix(construct):
    # ends that only combine to object (Fraction and float) are rejected
    with pytest.raises(TypeError, match="exactly"):
        construct()


def test_explicit_mix():
    # exact promotions and explicit subtypes are allowed
    assert pd.array([Interval(0, F(1, 2))], dtype="span").dtype == IntervalDType(
        FractionDType()
    )
    as_object = pd.array([Interval(F(0), 0.5)], dtype="span[object]")
    assert as_object[0] == Interval(F(0), 0.5)
    as_float = IntervalArray.from_arrays(frac([0]), [0.5], dtype="span[Float64]")
    assert as_float[0] == Interval(0, 0.5)


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


def spans(*ivs):
    return pd.array(list(ivs), dtype="span")


def test_shift():
    arr = spans(Interval(F(0), F(1)), None, Interval(F(1), F(3)))
    shifted = spans(Interval(F(1, 2), F(3, 2)), None, Interval(F(3, 2), F(7, 2)))
    assert list((arr + F(1, 2)).isna()) == [False, True, False]
    assert (arr + F(1, 2) == shifted).sum() == 2
    assert (F(1, 2) + arr == shifted).sum() == 2
    assert ((arr + F(1, 2)) - F(1, 2) == arr).sum() == 2
    assert ((arr + 1).duration == arr.duration).sum() == 2


def test_scale():
    arr = spans(Interval(F(1), F(3)), None)
    assert (arr * 2)[0] == Interval(2, 6)
    assert (2 * arr)[0] == Interval(2, 6)
    assert (arr * 0)[0] == Interval(0, 0)
    assert pd.isna((arr * 2)[1])
    with pytest.raises(ValueError):
        arr * -1


def test_elementwise():
    arr = spans(Interval(F(0), F(1)), Interval(F(0), F(1)), Interval(F(0), F(1)))
    result = arr + frac([0, F(1, 2), None])
    assert result[0] == Interval(0, 1)
    assert result[1] == Interval(F(1, 2), F(3, 2))
    assert pd.isna(result[2])
    # a list operand is inferred like the ends
    assert (arr * [1, F(1, 2), 2]).dtype == IntervalDType(FractionDType())


def test_arith_subtype():
    # Int64 and Fraction give fractions, fractions and floats give floats
    assert (spans(Interval(0, 1)) + F(1, 2)).dtype == IntervalDType(FractionDType())
    assert (spans(Interval(0, 1)) + F(1, 2))[0] == Interval(F(1, 2), F(3, 2))
    as_float = spans(Interval(F(0), F(1, 2))) * 0.5
    assert as_float.dtype == IntervalDType("Float64")
    assert as_float[0] == Interval(0, 0.25)
    # Series keep the span dtype
    ser = pd.Series(spans(Interval(F(0), F(1))))
    assert (ser + 1).dtype == ser.dtype
    assert (ser + 1)[0] == Interval(1, 2)


@pytest.mark.parametrize(
    "operation",
    [
        lambda a: 1 - a,  # would mirror the intervals
        lambda a: a / 2,
        lambda a: a // 1,
        lambda a: a % 1,
        lambda a: a**2,
        lambda a: a + a,
        lambda a: a + Interval(0, 1),
        lambda a: Interval(0, 1) + a,
        lambda a: a + [Interval(0, 1)],
        lambda a: pd.Series(a) * pd.Series(a),
    ],
)
def test_arith_unsupported(operation):
    with pytest.raises(TypeError):
        operation(spans(Interval(F(0), F(1))))


def test_map_time():
    arr = spans(Interval(F(1, 2), F(3, 2)), None)
    squared = arr.map_time(lambda x: x * x)
    assert squared[0] == Interval(F(1, 4), F(9, 4))
    assert pd.isna(squared[1])
    as_float = arr.map_time(lambda x: x.astype("Float64"))
    assert as_float.dtype == IntervalDType("Float64")
    with pytest.raises(ValueError):
        arr.map_time(lambda x: -x)


def test_min_max():
    ivs = [Interval(F(1), F(2)), None, Interval(F(0), F(3)), Interval(F(0), F(1))]
    ser = pd.Series(pd.array(ivs, dtype="span"))
    # by left end, then by right end, like sorting
    assert ser.min() == Interval(0, 1)
    assert ser.max() == Interval(1, 2)
    assert pd.isna(ser.min(skipna=False))
    assert pd.isna(pd.Series(pd.array([None], dtype="span[frac]")).max())
    groups = ser.groupby([1, 1, 1, 2])
    assert groups.min().tolist() == [Interval(0, 3), Interval(0, 1)]
    with pytest.raises(TypeError):
        ser.sum()
