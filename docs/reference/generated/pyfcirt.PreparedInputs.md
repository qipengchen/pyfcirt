# PreparedInputs

```{eval-rst}
.. autoclass:: pyfcirt.PreparedInputs
```

Read the prepared identities and arrays shared by construction and analysis.

## Parameters


`data` : xarray.DataTree
: Validated prepared input tree. Use `PreparedInputs.from_inputs()` to create
  it from design and response declarations.

`_response_projection` : mapping or None
: Internal cache input. The public preparation method supplies it when needed.


## Returns

PreparedInputs
: Read-only runtime inputs consumed by `Model`. Key accessors are:

  - `coords`: dimension names and labels such as `person`, `item` and `trait`.
  - `constant_data`: design, parameter identities and observation catalogue.
  - `observed_data`: recorded answers and administration state.
  - `parameter_layouts`: final parameter-entry layouts by parameter kind.
  - `output_coords`: extra identity coordinates for parameter and observation axes.


## Notes

`annotate_coordinates(dataset, *, require=False)` attaches auxiliary labels using the dataset's
existing axes and returns a dataset in the same order. Unknown primary labels
and conflicting auxiliary coordinates raise `ValueError`. With `require=True`,
all applicable auxiliary coordinates must already be present. Use
`dataset.copy(deep=True)` before editing prepared data.

## Examples

```python
import pyfcirt as fc

example = fc.load_example("simulated_mirt", model="DichotomousIRT", n_persons=3)
prepared = fc.PreparedInputs.from_inputs(example.data)
print({axis: prepared.constant_data.sizes[axis] for axis in ("person", "item", "trait")})
```

```text
{'person': 3, 'item': 8, 'trait': 1}
```

## See also

[PreparedInputs.from_inputs](pyfcirt.PreparedInputs.from_inputs.md) and the [coordinate contract](../parameter_contract.md).
