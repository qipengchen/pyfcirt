# FCDesignMapping

```{eval-rst}
.. autoclass:: pyfcirt.FCDesignMapping
```

Declare forced-choice membership with dictionaries instead of a table or design string.

## Parameters


`traits` : mapping
: Map each trait ID to its statement IDs. Assign one trait per statement.

`blocks` : mapping
: Map each block ID to its statement IDs in presentation order.

`keying` : mapping or None
: Map statement IDs to loading signs (+1 or −1). Omitted signs default to +1.


## Returns

FCDesignMapping
: Design for `FCInput.design`. Block-list order determines the positions
  referenced by a rank, pick or most–least response.

## Examples

```python
import pyfcirt as fc

design = fc.FCDesignMapping(
    traits={"agency": ["A", "C"], "social": ["B"]},
    blocks={"B1": ["A", "B", "C"]},
    keying={"C": -1},
)
print(design.blocks["B1"])
```

```text
['A', 'B', 'C']
```

## See also

[FCInput](pyfcirt.FCInput.md)
