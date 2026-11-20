# ItemDesignMapping

```{eval-rst}
.. autoclass:: pyfcirt.ItemDesignMapping
```

Declare item-to-trait membership with a dictionary.

## Parameters


`traits` : mapping
: Map each trait ID to its item IDs. Listing an item under multiple traits
  gives it an active loading on each of those traits.


## Returns

ItemDesignMapping
: Design for `ItemInput.design`.

## Examples

```python
import pandas as pd
import pyfcirt as fc

design = fc.ItemDesignMapping(traits={"anxiety": ["I1", "I2"]})
answers = fc.ResponseTable(
    pd.DataFrame({"person_id": ["P1", "P1"], "item_id": ["I1", "I2"], "response": [1, 0]}),
    format="item_long",
)
prepared = fc.PreparedInputs.from_inputs(fc.ItemInput(design, answers))
print(prepared.coords["item"].tolist())
```

```text
['I1', 'I2']
```

## See also

[ItemInput](pyfcirt.ItemInput.md)
