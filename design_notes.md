# Design Notes

### Immutable Values

`Interval` is a frozen dataclass:

- Intervals are immutable and hashable, so they can be used as dictionary or groupby keys.
- Intervals are sorted by `(left, right)`, the natural order of events.
- Equality and hashing are consistent across numeric types,
  since Python's numbers compare and hash consistently:
  `Interval(1, 2) == Interval(1.0, Fraction(2))`.


## Interval Arrays

The reductions `min` and `max` (also in groupby) use the ordering of scalars,
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
