# BlockSpec

```{eval-rst}
.. autoclass:: pyfcirt.BlockSpec
```

Change the measurement model, response format or scale for one block.

## Parameters


`block_id` : str or int
: Block's ID in the design.

`model` : str, measurement specification or None
: Local response model. `None` inherits the design or input default.

`response_format` : {"rank", "pick", "mole"}, None or Ellipsis
: Local task. The default `...` inherits. Explicit `None` declares
  separate pair judgments.

`scale` : CategoryScale, UnitIntervalScale or None
: Local preference scale. `None` inherits.


## Returns

BlockSpec
: One override in `FCInput(..., blocks=(override,))`. Other blocks keep their
  settings.

## Examples

Declare B4 as a TwoPLRank block that records the most and least preferred statements.

```python
import pyfcirt as fc

override = fc.BlockSpec("B4", model="TwoPLRank", response_format="mole")
print(override.block_id, override.model, override.response_format)
```

```text
B4 TwoPLRank mole
```

## See also

[Forced-choice declarations](pyfcirt.FCInput.md)
