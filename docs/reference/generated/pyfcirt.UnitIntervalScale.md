# UnitIntervalScale

```{eval-rst}
.. autoclass:: pyfcirt.UnitIntervalScale
```

Declare a continuous preference score on the open interval (0, 1).

## Parameters


`higher_prefers` : {"first", "second"}
: Which recorded statement is favored by a larger value. Default: `"first"`.


## Returns

UnitIntervalScale
: Continuous pair scale for `FCInput(..., response_format=None, scale=scale)`
  or `BlockSpec(..., scale=scale)`.

## Notes

Use a Beta link in TIRT or GTUM. Recorded values must be strictly between
0 and 1. A value closer to 1 indicates a stronger preference for the
statement named by `higher_prefers`.

## Examples

Use this direction when a larger response favors the second statement.

```python
import pyfcirt as fc

scale = fc.UnitIntervalScale(higher_prefers="second")
print(scale.higher_prefers)
```

```text
second
```

## See also

[TIRT](pyfcirt.TIRT.md) and [GTUM](pyfcirt.GTUM.md).
