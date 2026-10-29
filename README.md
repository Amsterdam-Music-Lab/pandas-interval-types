# pandas-interval-types

**Note**: This is experimental and almost entirely AI generated.
Use with caution but feel free to report bugs or suggest improvements.

---

This library implements time intervals as they are used to represent events in music.
Intervals consist of a left and right end and normally represent the left-closed timespan `[left, right)`.
Time points can be defined using arbitrary time types, such as `int`, `float`, or `Fraction`.

```python
from fractions import Fraction
from pandas_interval_types import Interval

iv = Interval(Fraction(1, 2), Fraction(3, 2))
iv.duration         # Fraction(1, 1)
Fraction(1) in iv   # True
iv + 1              # Interval(left=Fraction(3, 2), right=Fraction(5, 2))
2 * iv              # Interval(left=Fraction(1, 1), right=Fraction(3, 1))
iv * Fraction(1, 2) # Interval(left=Fraction(1, 4), right=Fraction(3, 4))
iv.map_time(float)  # Interval(left=0.5, right=1.5)
```

pandas' own `Interval` doesn't support exact time types:
`pd.Interval(Fraction(0), Fraction(1))` raises a `ValueError`
("Only numeric, Timestamp and Timedelta endpoints are allowed").
This package provides the scalar `Interval` class
and a pandas extension array of intervals, `IntervalArray`, with the dtype `span[<subtype>]`:

```python
import pandas as pd
import pandas_fractions
from pandas_interval_types import IntervalArray

onsets = pd.array([0, Fraction(1, 2)], dtype="frac")
spans = pd.Series(IntervalArray.from_arrays(onsets, onsets + Fraction(1, 2)))
spans.dtype           # span[frac]
spans[1]              # Interval(left=Fraction(1, 2), right=Fraction(1, 1))
spans.array.duration  # FractionArray [1/2, 1/2]
pd.Series([Interval(0, Fraction(1, 2))], dtype="span")  # infers span[frac]
```

## Design

### Generic Time Type

`Interval[T: Time]` is generic over the type of its ends.
`Time` is a `Protocol` that lists what an interval needs from its time type:
ordering (`<`, `<=`) and arithmetic (`+`, `-`, `*`).
Any type that provides these works, e.g. `int`, `float`, `Fraction`, or `Decimal`.
The type parameter only matters for type checkers; at runtime, the class is duck-typed.

Durations, shifts, and scaling factors have the same type as the time points.
This holds for numbers, but not for types like `datetime`/`timedelta`,
which would need separate type parameters.
Due to duck-typed time types, this would still work at runtime.

### Left-Closed Intervals

An interval contains all time points `t` with `left <= t < right`.
All intervals are left-closed, so there is no `closed` attribute.
This way, consecutive events (e.g. notes) meet without overlapping.
An interval whose right end is before its left end raises a `ValueError`.

The ends are called `left` and `right`, like in `pd.Interval`,
so code that only reads `.left` and `.right` works with both.
`left` is the onset and `right` the offset of an interval.

#### Instantaneous Intervals

The special case where `left == right` is interpreted as the closed interval `[left,right]`, containing exactly one time point `t == left == right`.
The interval relations and operations here are extended to handle this case accordingly.
This is done for two reasons:
1. Operations on inexact time types such as `float` might produce empty intervals due to rounding errors.
   In this case, it makes more sense to assume that the interval is not empty but infinitesimally small.
   The user can then decide if empty intervals should be discarded or not.
   This can also be useful in the case of quantization to prevent collapsing intervals from vanishing.
2. Empty intervals can be useful to represent events that logically have a duration of 0 but are realized by a non-zero duration in performance (such as grace notes).

In either case, the user (or a downstream library) can still decide to restrict intervals to postive duration and discard intervals with 0 duration, depending on the use case.


### Operations

`Interval` only has operations on a single interval (and a scalar).
Relations and operations between intervals (e.g. "overlaps", intersection)
are meant to be separate functions written against `.left` and `.right`,
so that they work for scalar intervals and arrays alike.

- `duration` is `right - left`.
- `t in iv` checks whether the time point `t` is in the interval.
- `iv.map_time(f)` applies `f` to both ends.
  It can change the time type (`iv.map_time(float)`) or transform time (e.g. a tempo map).
  `f` must be non-decreasing:
  a decreasing function would swap the ends,
  and the mirrored interval would be closed on the right,
  which an `Interval` can't represent.
  The check in the constructor turns this into a `ValueError`.
- Arithmetic is defined via `map_time`:
  - `iv + t`, `t + iv`, and `iv - t` shift the interval (the duration stays the same).
  - `iv * s` and `s * iv` scale the interval around 0 (the duration is scaled by `s`).
    The factor must not be negative, for the same reason as above.
    A factor of 0 gives a zero-length interval.
  - There is no division: it can be expressed as multiplication with the reciprocal
    (`iv * Fraction(1, 2)`) for types that have one,
    and it would change the time type for others (`int / int` is a `float`),
    which wouldn't fit `Interval[T]`.
  - Left out on purpose:
    - `t - iv` (it would mirror the interval),
    - Operations between two intervals:
      Since the time type is duck-typed, the operators explicitly reject `Interval` operands,
    otherwise `iv1 + iv2` would produce an interval with intervals as its ends.
        The same goes for collections like arrays, so that `iv + array` is left to the array),
    - `//` and `%`:
      Rounding both ends independently isn't a meaningful interval operation.
      Use `iv.map_time(math.floor)` if needed.

## Pandas Extension Array

A pandas `ExtensionDtype` and `ExtensionArray` are provided for intervals to make it easier to work with them in dataframes.

### Storage

An `IntervalArray` stores the ends of its intervals as two pandas extension arrays,
`left` and `right`, of the same dtype, the subtype.
For example, the ends of a `span[frac]` array are two `FractionArray`s.
Like in the scalar class, the subtype can be anything with ordering and arithmetic,
and all operations on intervals are written against the operations of the ends' arrays.

The ends are always extension arrays, so that they can represent missing values.
numpy subtypes are replaced by the extension dtypes pandas uses for them,
e.g. `span[int64]` becomes `span[Int64]` and `span[float64]` becomes `span[Float64]`,
and other numpy dtypes (like `object`) are wrapped.
Otherwise, inserting a missing value would change the dtype of the ends (e.g. from int to float).
An interval is missing if both of its ends are missing,
and the ends must be missing at the same positions.

### The dtype

The dtype is called `span[<subtype>]`, e.g. `span[frac]` or `span[Int64]`,
because pandas' own `IntervalDtype` claims every dtype string that contains `interval[`.
Without a subtype, `span` infers the subtype from the data.
Both ends are inferred together, so they always get the same subtype.
If pandas only infers `object`, as for `Fraction`s,
the registered extension dtypes for the type of the ends are tried (e.g. `frac` for `Fraction`).
Ends given as object arrays are inferred the same way,
e.g. `from_arrays` with two object Series of `Fraction`s gives `span[frac]`.
Types without a dtype of their own (e.g. `Decimal`) give `span[object]`.

Ends of different types are combined if this is exact,
e.g. `int`s and `Fraction`s give `frac`, and `int`s and floats give `Float64`.
Types that only combine to `object`, such as `Fraction`s and floats, raise a `TypeError`,
whether they are mixed within an interval, across intervals, or between the two ends.
Mixing them is usually a mistake (e.g. by using different time units),
and `span[object]` would be easy to miss.
To combine them anyway, convert them explicitly to a common type (e.g. with `astype` or `map_time`),
or ask for an explicit subtype like `span[Float64]` or `span[object]`,
which is used without inference.
(The scalar `Interval` follows Python's rules instead, see "Mixed time types".)

Arrays with different subtypes combine if their subtypes do,
e.g. concatenating `span[frac]` and `span[Int64]` gives `span[frac]`.
`astype("span[<subtype>]")` converts the ends, e.g. from `frac` to `Float64`.

### What is Supported


Like `Interval`, the array has `duration`, `map_time`, and arithmetic.

In addition, the array implements the `ExtensionArray` interface,
which makes indexing, assignment, missing values, concatenation,
sorting, grouping, factorizing, and comparisons work in pandas.
Comparisons and sorting use the lexicographic order on `(left, right)`, like `Interval`. The reductions `arr.min()` and `arr.max()` are supported, other reductions such as `arr.sum()` are not meaningful.

