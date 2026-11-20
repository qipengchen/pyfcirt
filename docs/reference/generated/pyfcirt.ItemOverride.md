# ItemOverride

```{eval-rst}
.. autoclass:: pyfcirt.ItemOverride
```

Assign a local measurement model or response scale to one individual item.

## Parameters


`item_id` : str or int
: Item's ID in the design.

`model` : str, measurement specification or None
: Local measurement model. `None` inherits.

`categories` : tuple of int or None
: Complete, increasing consecutive integer codes for the item. `None`
  inherits the input scale.


## Returns

ItemOverride
: One override in `ItemInput(..., items=(override,))`.

## Examples

Use a binary model and scale for item I3 in an otherwise ordinal questionnaire.

```python
import pyfcirt as fc

override = fc.ItemOverride("I3", model="DichotomousIRT", categories=(0, 1))
print(override.item_id, override.model, override.categories)
```

```text
I3 DichotomousIRT (0, 1)
```

## See also

[ItemInput example](pyfcirt.ItemInput.md#mix-ordinal-and-binary-items)
