# CategoryScale

```{eval-rst}
.. autoclass:: pyfcirt.CategoryScale
```

Declare every category of a binary or graded pair judgment.

## Parameters


`categories` : tuple
: Complete distinct response codes, from lowest to highest, including
  allowed codes absent from the data.

`higher_prefers` : {"first", "second"}
: Which statement is favored by a higher category. Default: `"first"`.

`labels` : tuple or None
: Optional display labels, one per category in the same order. `None` uses
  category codes.


## Returns

CategoryScale
: Discrete pair scale to pass to `FCInput(..., response_format=None, scale=scale)`
  or to a `BlockSpec`.

## Notes

The direction is relative to the recorded `statement_1` and `statement_2`.
Statement keying describes the direction of a trait loading.

## Examples

```python
import pyfcirt as fc

scale = fc.CategoryScale(
    (1, 2, 3),
    labels=("prefer second", "tie", "prefer first"),
)
print(dict(zip(scale.categories, scale.labels, strict=True)))
```

```text
{1: 'prefer second', 2: 'tie', 3: 'prefer first'}
```

## See also

[UnitIntervalScale](pyfcirt.UnitIntervalScale.md) for continuous choices.
