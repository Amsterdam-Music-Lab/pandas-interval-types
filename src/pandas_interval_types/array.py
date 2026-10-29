import operator
import re

import numpy as np
import pandas as pd
from pandas._typing import type_t
from pandas.api.extensions import ExtensionArray, ExtensionDtype, no_default
from pandas.core import arraylike, roperator
from pandas.core.dtypes.base import _registry
from pandas.core.dtypes.cast import find_common_type

from pandas_interval_types.interval import Interval


def _canonical_subtype(subtype) -> ExtensionDtype:
    """
    Convert a dtype (or dtype string) to the dtype in which interval ends are stored.

    Ends are always stored as extension arrays, so that they can represent NA.
    numpy dtypes are replaced by the extension dtypes pandas uses for them,
    e.g. int64 by the nullable Int64, so that taking with NA doesn't change the dtype.
    """
    subtype = pd.api.types.pandas_dtype(subtype)
    if isinstance(subtype, np.dtype):
        return pd.array(np.empty(0, dtype=subtype)).dtype
    return subtype


@pd.api.extensions.register_extension_dtype
class IntervalDType(ExtensionDtype):
    """
    A pandas dtype for left-closed intervals, parametrized by the type of their ends.

    The string name is "span[<subtype>]", e.g. "span[frac]" or "span[Int64]",
    or just "span" for an unspecified subtype, which is inferred from the data.
    (pandas' own IntervalDtype claims every name containing "interval[".)
    """

    type = Interval
    _metadata = ("subtype",)
    _match = re.compile(r"^span(\[(?P<subtype>.+)\])?$")

    def __init__(self, subtype=None):
        self.subtype = None if subtype is None else _canonical_subtype(subtype)

    @property
    def name(self) -> str:
        return "span" if self.subtype is None else f"span[{self.subtype}]"

    @classmethod
    def construct_from_string(cls, string: str) -> "IntervalDType":
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        match = cls._match.match(string)
        if match is None:
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")
        return cls(match["subtype"])

    @classmethod
    def construct_array_type(cls) -> type_t["IntervalArray"]:
        return IntervalArray

    def _get_common_dtype(self, dtypes):
        # intervals combine if their subtypes do (e.g. frac and Int64 give frac)
        if not all(isinstance(t, IntervalDType) for t in dtypes):
            return None
        if any(t.subtype is None for t in dtypes):
            return None
        subtype = find_common_type([t.subtype for t in dtypes])
        if pd.api.types.is_object_dtype(subtype):
            return None
        return IntervalDType(subtype)


def _to_bool(values) -> np.ndarray:
    """Convert a boolean array to numpy, with NA as False."""
    if isinstance(values, ExtensionArray):
        return values.to_numpy(dtype=bool, na_value=False)
    return np.asarray(values, dtype=bool)


def _infer_ends(values: list) -> ExtensionArray:
    """
    Convert a list of interval ends (with None for NA) to an extension array.

    Like pd.array, but if pandas only infers object, this tries the registered
    extension dtypes for the type of the values, e.g. frac for Fraction.
    """
    # pandas' inference handles builtin types, e.g. ints become Int64
    array = pd.array(values)
    if not pd.api.types.is_object_dtype(array.dtype):
        return _canonical_array(array)
    # object: look for registered dtypes whose scalar type occurs in the values
    valid = [x for x in values if not (pd.api.types.is_scalar(x) and pd.isna(x))]
    types = {type(x) for x in valid}
    for dtype_class in _registry.dtypes:
        if getattr(dtype_class, "type", None) not in types:
            continue
        try:
            # dtypes that need parameters can't be constructed and are skipped
            dtype = dtype_class()
            # _from_scalars is strict, e.g. FractionArray doesn't convert floats
            return dtype.construct_array_type()._from_scalars(values, dtype=dtype)
        except TypeError, ValueError:
            continue
    # no dtype accepts all values, so keep them as objects,
    # unless they are a lossy mix of types that have dtypes on their own
    if len(types) > 1:
        dtypes = [_infer_ends([x for x in valid if type(x) is t]).dtype for t in types]
        if not any(pd.api.types.is_object_dtype(t) for t in dtypes):
            _check_exact_mix(dtypes, [t.__name__ for t in types])
    return array


def _check_exact_mix(dtypes, names) -> None:
    """Raise a TypeError if the dtypes of interval ends only combine to object."""
    if pd.api.types.is_object_dtype(find_common_type(dtypes)):
        raise TypeError(
            f"interval ends of types {' and '.join(sorted(names))} "
            "can't be combined exactly, convert them explicitly (e.g. with astype)"
        )


def _canonical_array(values) -> ExtensionArray:
    """Convert array-like interval ends to an extension array of a canonical subtype."""
    if isinstance(values, (pd.Series, pd.Index)):
        values = values.array
    if isinstance(values, ExtensionArray):
        if isinstance(values, pd.arrays.NumpyExtensionArray):
            # a wrapped numpy array, which is canonicalized like numpy arrays below
            values = values.to_numpy()
        else:
            # other extension arrays (e.g. FractionArray, Int64) are kept as they are
            return values
    if isinstance(values, np.ndarray):
        if values.dtype == np.dtype(object):
            # objects may have a more specific dtype, e.g. Fractions (as in a Series)
            return _infer_ends(list(values))
        # e.g. int64 becomes Int64, which can hold NA
        return pd.array(values, dtype=_canonical_subtype(values.dtype))
    # other sequences (e.g. lists) have no dtype, so it is inferred
    return _infer_ends(list(values))


def _as_subtype(values, subtype) -> ExtensionArray:
    """Convert array-like interval ends to the given (canonical) subtype."""
    if isinstance(values, (pd.Series, pd.Index)):
        values = values.array
    if isinstance(values, ExtensionArray):
        return values.astype(subtype, copy=False)
    return pd.array(values, dtype=subtype)


class IntervalArray(arraylike.OpsMixin, ExtensionArray):
    """
    An array of left-closed intervals [left, right).

    The ends are stored as two extension arrays of the same dtype (the subtype),
    e.g. two FractionArrays for exact times.
    A missing interval has both ends missing:
    the ends are always missing at the same positions (checked by from_arrays).
    """

    def __init__(self, left: ExtensionArray, right: ExtensionArray):
        # unchecked, use from_arrays to validate the ends
        self._left = left
        self._right = right

    @classmethod
    def from_arrays(cls, left, right, dtype=None) -> "IntervalArray":
        """
        Create intervals from arrays of their left and right ends.

        The ends are converted to a common subtype, or to the subtype of `dtype`.
        Each interval must satisfy left <= right,
        and left must be missing exactly where right is missing.
        """
        dtype = IntervalDType() if dtype is None else pd.api.types.pandas_dtype(dtype)
        if not isinstance(dtype, IntervalDType):
            raise TypeError(f"{dtype} is not an interval dtype")
        subtype = dtype.subtype
        if subtype is None:
            # infer the ends' dtypes and combine them, if this is exact
            left, right = _canonical_array(left), _canonical_array(right)
            if left.dtype != right.dtype:
                _check_exact_mix(
                    [left.dtype, right.dtype], [str(left.dtype), str(right.dtype)]
                )
            subtype = _canonical_subtype(find_common_type([left.dtype, right.dtype]))
        # an explicit subtype is used without inference, e.g. object accepts any mix
        left, right = _as_subtype(left, subtype), _as_subtype(right, subtype)
        if len(left) != len(right):
            raise ValueError("left and right must have the same length")

        na = left.isna()
        if np.any(na != right.isna()):
            raise ValueError("left and right must be missing at the same positions")
        before = _to_bool(right < left) & ~na
        if np.any(before):
            i = np.argmax(before)
            raise ValueError(f"right end {right[i]} is before left end {left[i]}")
        return cls(left, right)

    @classmethod
    def from_breaks(cls, breaks, dtype=None) -> "IntervalArray":
        """
        Create the intervals between consecutive break points.

        n breaks give n - 1 intervals [breaks[i], breaks[i + 1]).
        The breaks are converted like the ends in from_arrays,
        must not be missing, and must be non-decreasing.
        """
        dtype = IntervalDType() if dtype is None else pd.api.types.pandas_dtype(dtype)
        if not isinstance(dtype, IntervalDType):
            raise TypeError(f"{dtype} is not an interval dtype")
        # convert all breaks together, so that both ends get the same subtype
        if dtype.subtype is None:
            breaks = _canonical_array(breaks)
        else:
            breaks = _as_subtype(breaks, dtype.subtype)
        if np.any(breaks.isna()):
            raise ValueError("breaks must not be missing")
        return cls.from_arrays(breaks[:-1], breaks[1:], dtype=dtype)

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        if isinstance(scalars, cls) and (dtype is None or dtype == scalars.dtype):
            return scalars.copy() if copy else scalars
        lefts, rights = [], []
        for x in scalars:
            if isinstance(x, Interval):
                lefts.append(x.left)
                rights.append(x.right)
            elif pd.api.types.is_scalar(x) and pd.isna(x):
                lefts.append(None)
                rights.append(None)
            else:
                raise TypeError(f"{type(x).__name__} is not an Interval")
        if dtype is not None:
            dtype = pd.api.types.pandas_dtype(dtype)
        subtype = getattr(dtype, "subtype", None)
        if subtype is None:
            # infer the subtype from both ends together, so that they agree
            ends = _infer_ends(lefts + rights)
            n = len(lefts)
            left, right = ends[:n], ends[n:]
        else:
            left = pd.array(lefts, dtype=subtype)
            right = pd.array(rights, dtype=subtype)
        return cls.from_arrays(left, right, dtype=dtype)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls._from_sequence(values, dtype=original.dtype)

    # the ends
    # --------

    @property
    def left(self) -> ExtensionArray:
        return self._left

    @property
    def right(self) -> ExtensionArray:
        return self._right

    @property
    def duration(self) -> ExtensionArray:
        return self._right - self._left

    def map_time(self, f) -> "IntervalArray":
        """
        Apply f to the arrays of left and right ends.

        f takes and returns an array of time values (e.g. a FractionArray),
        and must be non-decreasing elementwise, like for Interval.map_time.
        Otherwise, the ends would be swapped, which raises a ValueError.
        """
        return type(self).from_arrays(f(self._left), f(self._right))

    # arithmetic
    # ----------
    # Like Interval: shifting (+, -) and scaling (*) by time values,
    # either scalars or arrays of the same length (elementwise).

    _ARITH_OPS = (
        operator.add,
        roperator.radd,
        operator.sub,
        operator.mul,
        roperator.rmul,
    )

    def _arith_method(self, other, op):
        if op not in self._ARITH_OPS:
            return NotImplemented
        if isinstance(other, Interval):
            other = None
        elif pd.api.types.is_scalar(other):
            # Convert scalars to arrays, so that the arrays of the ends and the operand
            # decide the result type between them. E.g. an Int64 array computes
            # Int64 + Fraction on its int64 data, which gives objects,
            # but defers Int64 + FractionArray to FractionArray, which gives fractions.
            other = _infer_ends([other] * len(self))
        elif not isinstance(other, IntervalArray):
            # e.g. a list of Fractions becomes a FractionArray
            other = _canonical_array(other)
            # a list of intervals stays an object array
            if pd.api.types.is_object_dtype(other.dtype) and any(
                isinstance(x, Interval) for x in other
            ):
                other = None
        if other is None or isinstance(other, IntervalArray):
            # Raised instead of returning NotImplemented,
            # so that Python doesn't fall back to Interval.__radd__ etc.
            raise TypeError(
                "intervals can't be operands of interval arithmetic, "
                "use separate functions for operations between intervals"
            )
        return self.map_time(lambda ends: op(ends, other))

    # the ExtensionArray interface
    # ----------------------------

    @property
    def dtype(self) -> IntervalDType:
        return IntervalDType(self._left.dtype)

    def __len__(self) -> int:
        return len(self._left)

    def __getitem__(self, item):
        if pd.api.types.is_integer(item):
            if pd.isna(self._left[item]):
                return self.dtype.na_value
            return Interval(self._left[item], self._right[item])
        item = pd.api.indexers.check_array_indexer(self, item)
        return type(self)(self._left[item], self._right[item])

    def __setitem__(self, key, value):
        if self._readonly:
            raise ValueError("Cannot modify read-only array")
        key = pd.api.indexers.check_array_indexer(self, key)
        scalar = isinstance(value, Interval) or (
            pd.api.types.is_scalar(value) and pd.isna(value)
        )
        # convert the value before assigning,
        # so that an invalid value can't leave the array half-modified
        value = type(self)._from_sequence(
            [value] if scalar else value, dtype=self.dtype
        )
        left, right = value._left, value._right
        if scalar:
            # assign the ends themselves, which the ends' arrays broadcast to any key
            left, right = left[0], right[0]
        self._left[key] = left
        self._right[key] = right

    @property
    def _readonly(self) -> bool:
        return self._left._readonly

    @_readonly.setter
    def _readonly(self, value: bool):
        self._left._readonly = value
        self._right._readonly = value

    @property
    def nbytes(self) -> int:
        return self._left.nbytes + self._right.nbytes

    def isna(self) -> np.ndarray:
        # the ends are missing at the same positions, so checking one is enough
        return np.asarray(self._left.isna())

    def take(self, indices, *, allow_fill=False, fill_value=None):
        if allow_fill and not (
            pd.api.types.is_scalar(fill_value) and pd.isna(fill_value)
        ):
            fill = type(self)._from_sequence([fill_value], dtype=self.dtype)
            fill_left, fill_right = fill._left[0], fill._right[0]
        else:
            fill_left = fill_right = None
        left = self._left.take(indices, allow_fill=allow_fill, fill_value=fill_left)
        right = self._right.take(indices, allow_fill=allow_fill, fill_value=fill_right)
        return type(self)(left, right)

    def copy(self):
        return type(self)(self._left.copy(), self._right.copy())

    @classmethod
    def _concat_same_type(cls, to_concat):
        left_type = type(to_concat[0]._left)
        return cls(
            left_type._concat_same_type([a._left for a in to_concat]),
            left_type._concat_same_type([a._right for a in to_concat]),
        )

    # conversion
    # ----------

    def to_numpy(self, dtype=None, copy=False, na_value=no_default):
        # the conversion always creates a new array, so `copy` doesn't matter
        result = np.empty(len(self), dtype=object)
        result[:] = [self[i] for i in range(len(self))]
        if na_value is not no_default:
            result[self.isna()] = na_value
        return result if dtype is None else result.astype(dtype)

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError(
                "An IntervalArray can't be converted to numpy without a copy"
            )
        return self.to_numpy(dtype=dtype)

    def astype(self, dtype, copy=True):
        dtype = pd.api.types.pandas_dtype(dtype)
        if isinstance(dtype, IntervalDType):
            if dtype.subtype is None or dtype == self.dtype:
                return self.copy() if copy else self
            # convert the ends, e.g. from frac to Float64
            return type(self).from_arrays(
                self._left.astype(dtype.subtype), self._right.astype(dtype.subtype)
            )
        return super().astype(dtype, copy=copy)

    def _values_for_json(self) -> np.ndarray:
        # objects like {"left": "1/2", "right": "1"}, with the ends converted like their arrays
        left, right = self._left._values_for_json(), self._right._values_for_json()
        values = [
            None if na else {"left": l, "right": r}
            for na, l, r in zip(self.isna(), left, right)
        ]
        result = np.empty(len(self), dtype=object)
        result[:] = values
        return result

    # sorting and factorizing
    # -----------------------
    # Factorizing uses the default implementation (with Interval objects).

    def _values_for_argsort(self) -> np.ndarray:
        """
        Return integer ranks that sort like the intervals, by (left, right).

        Equal intervals get equal ranks, missing values the highest rank.
        """
        na = self.isna()
        valid = self[~na]
        # sort by right first, then stably by left
        order = valid._right.argsort(kind="stable")
        order = order[valid._left.take(order).argsort(kind="stable")]
        left, right = valid._left.take(order), valid._right.take(order)
        new_value = np.ones(len(order), dtype=bool)
        new_value[1:] = ~(
            _to_bool(left[1:] == left[:-1]) & _to_bool(right[1:] == right[:-1])
        )
        valid_ranks = np.empty(len(order), dtype=np.int64)
        valid_ranks[order] = np.cumsum(new_value) - 1
        ranks = np.full(len(self), len(order), dtype=np.int64)
        ranks[~na] = valid_ranks
        return ranks

    # reductions
    # ----------

    def _reduce(self, name: str, *, skipna=True, keepdims=False, **kwargs):
        if name not in ("min", "max"):
            # raises a TypeError
            return super()._reduce(name, skipna=skipna, keepdims=keepdims, **kwargs)
        # the earliest or latest interval, in the same order as for sorting
        na = self.isna()
        if na.all() or (na.any() and not skipna):
            index = -1  # NA with allow_fill
        else:
            ranks = self._values_for_argsort()  # NA has the highest rank
            if name == "min":
                index = int(np.argmin(ranks))
            else:
                index = int(np.argmax(np.where(na, -1, ranks)))
        result = self.take([index], allow_fill=True)
        return result if keepdims else result[0]

    # comparison
    # ----------

    def _cmp_method(self, other, op):
        if isinstance(other, Interval):
            other_left, other_right = other.left, other.right
            other_na = False
        elif pd.api.types.is_scalar(other) and pd.isna(other):
            return np.full(len(self), op is operator.ne)
        else:
            try:
                other = type(self)._from_sequence(other)
            except TypeError:
                other = None
            if other is None:
                # incomparable values are never equal
                if op is operator.eq:
                    return np.zeros(len(self), dtype=bool)
                if op is operator.ne:
                    return np.ones(len(self), dtype=bool)
                raise TypeError(f"Cannot compare {self.dtype} with non-intervals")
            if len(other) != len(self):
                raise ValueError("Lengths must match")
            other_left, other_right = other._left, other._right
            other_na = other.isna()

        same_left = _to_bool(self._left == other_left)
        if op in (operator.eq, operator.ne):
            equal = same_left & _to_bool(self._right == other_right)
            result = equal if op is operator.eq else ~equal
        else:
            # lexicographic order on (left, right), like Interval
            strict = operator.lt if op in (operator.lt, operator.le) else operator.gt
            result = _to_bool(strict(self._left, other_left)) | (
                same_left & _to_bool(op(self._right, other_right))
            )
        # NA behaves like NaN: only != is true
        return np.where(self.isna() | other_na, op is operator.ne, result)
