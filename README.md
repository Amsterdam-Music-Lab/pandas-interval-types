# pandas_intervals

Left-closed intervals `[left, right)` over arbitrary time types, such as `int`, `float`, or `Fraction`.

```python
from fractions import Fraction
from pandas_intervals import Interval

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
So far, this package contains only the scalar `Interval` class;
a vectorized pandas extension array may follow.

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
    otherwise `iv1 + iv2` would produce an interval with intervals as its ends),
    and `//` and `%` (rounding both ends independently isn't a meaningful interval operation;
    use `iv.map_time(math.floor)` if needed).

### Mixed time types

The ends of an interval can have different types,
and arithmetic follows Python's rules for mixing them.
`int` and `Fraction` mix without losing exactness (`Interval(0, Fraction(1, 2)) + 1`),
but `Fraction` and `float` silently give floats (`Interval(Fraction(0), Fraction(1, 2)) * 0.5`).
`Interval` doesn't check this, so keep time types consistent where exactness matters.
