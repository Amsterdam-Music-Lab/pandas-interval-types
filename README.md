# pandas-interval-types

**Note**: This is experimental and almost entirely AI generated.
Use with caution but feel free to report bugs or suggest improvements.

---

Left-closed intervals `[left, right)` over arbitrary time types, such as `int`, `float`, or `Fraction`.

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

### Generic time type

`Interval[T: Time]` is generic over the type of its ends.
`Time` is a `Protocol` that lists what an interval needs from its time type:
ordering (`<`, `<=`) and arithmetic (`+`, `-`, `*`).
Any type that provides these works, e.g. `int`, `float`, `Fraction`, or `Decimal`.
The type parameter only matters for type checkers; at runtime, the class is duck-typed.

Durations, shifts, and scaling factors have the same type as the time points.
This holds for numbers, but not for types like `datetime`/`timedelta`,
which would need separate type parameters.

### Left-closed intervals

An interval contains all time points `t` with `left <= t < right`.
All intervals are left-closed, so there is no `closed` attribute.
This way, consecutive events (e.g. notes) meet without overlapping.
Zero-length intervals (`left == right`) are allowed, but contain no time point.
An interval whose right end is before its left end raises a `ValueError`.

The ends are called `left` and `right`, like in `pd.Interval`,
so code that only reads `.left` and `.right` works with both.
`left` is the onset and `right` the offset of an interval.

### Immutable values

`Interval` is a frozen dataclass:

- Intervals are immutable and hashable, so they can be used as dictionary or groupby keys.
- Intervals are sorted by `(left, right)`, the natural order of events.
- Equality and hashing are consistent across numeric types,
  since Python's numbers compare and hash consistently:
  `Interval(1, 2) == Interval(1.0, Fraction(2))`.

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
    `t - iv` (it would mirror the interval),
    operations between two intervals (see above;
    since the time type is duck-typed, the operators explicitly reject `Interval` operands,
    otherwise `iv1 + iv2` would produce an interval with intervals as its ends;
    the same goes for collections like arrays, so that `iv + array` is left to the array),
    and `//` and `%` (rounding both ends independently isn't a meaningful interval operation;
    use `iv.map_time(math.floor)` if needed).

### Mixed time types

The ends of an interval can have different types,
and arithmetic follows Python's rules for mixing them.
`int` and `Fraction` mix without losing exactness (`Interval(0, Fraction(1, 2)) + 1`),
but `Fraction` and `float` silently give floats (`Interval(Fraction(0), Fraction(1, 2)) * 0.5`).
`Interval` doesn't check this, so keep time types consistent where exactness matters.

## The extension array

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
Mixing them is usually a mistake (e.g. score time and seconds),
and `span[object]` would be easy to miss.
To combine them anyway, convert them explicitly (e.g. with `astype` or `map_time`),
or ask for an explicit subtype like `span[Float64]` or `span[object]`,
which is used without inference.
(The scalar `Interval` follows Python's rules instead, see "Mixed time types".)

Arrays with different subtypes combine if their subtypes do,
e.g. concatenating `span[frac]` and `span[Int64]` gives `span[frac]`.
`astype("span[<subtype>]")` converts the ends, e.g. from `frac` to `Float64`.

### What is supported

The array implements the `ExtensionArray` interface,
which makes indexing, assignment, missing values, concatenation,
sorting, grouping, factorizing, and comparisons work in pandas.
Comparisons and sorting use the lexicographic order on `(left, right)`, like `Interval`.
The reductions `min` and `max` (also in groupby) use the same order,
so `ser.min()` is the earliest interval, and the shortest among those starting first.
Other reductions like `sum` or `mean` aren't meaningful for intervals and raise a `TypeError`
(for durations, use e.g. `arr.duration.sum()`).

Like `Interval`, the array has `duration`, `map_time`, and arithmetic:

- `arr.map_time(f)` applies `f` to the arrays of the ends (e.g. two `FractionArray`s),
  so `f` works on arrays: `arr.map_time(lambda x: x.astype("Float64"))`.
  As for `Interval`, `f` must be non-decreasing, otherwise it raises a `ValueError`.
- `arr + t`, `t + arr`, and `arr - t` shift, `arr * s` and `s * arr` scale the intervals.
  The operand is a time value or an array of time values of the same length (elementwise).
  A missing operand gives a missing interval.
  The same operations as for `Interval` are left out (division, `t - arr`, `//`, `%`, `**`),
  and intervals as operands raise a `TypeError`.
- The subtype of the result is the type of the ends' arithmetic,
  e.g. `span[Int64] + Fraction(1, 2)` gives `span[frac]`,
  and `span[frac] * 0.5` gives `span[Float64]`.
  To get this, scalar operands are first converted to arrays (like the ends are inferred),
  because pandas' `Int64` array would compute `Int64 + Fraction` on its int64 data,
  which gives objects, but defers `Int64 + FractionArray` to `FractionArray`.
  An operand that mixes `Fraction`s and floats raises a `TypeError`, like mixed ends.

It passes pandas' extension test suite, except for two cases that are unsupported by design:

- `pd.DataFrame([interval, ...])` turns each interval into a row with the columns
  `left` and `right`, because pandas treats lists of dataclasses as records.
  Use `pd.Series` or `pd.array` with the dtype instead.
- A JSON round trip doesn't restore intervals: they are written as objects like
  `{"left": "1/2", "right": "1"}` but read back as dicts.
