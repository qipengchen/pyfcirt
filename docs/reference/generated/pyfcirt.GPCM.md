# GPCM

```{eval-rst}
.. autoclass:: pyfcirt.GPCM
```

Generalized partial credit for ordered item responses.

## Parameters

`priors` : mapping or None
: Override local prior roles with unnamed PyMC distributions such as
  `pm.Normal.dist(mu=0, sigma=1)`. `None` uses the defaults below.


## Returns

GPCM
: A measurement declaration for `ItemInput(..., model=declaration)`. Pass the configured
  declaration as the `model` value when loading a bundled dataset.

## Notes

Give `ItemInput` all allowed category codes in order. The first category
 intercept is fixed at zero. Other category intercepts are estimated freely.

Default priors for active roles:

| Prior role | Default law |
| --- | --- |
| `loading_magnitude` | LogNormal(mu=0.25, sigma=0.5) |
| `category_intercept` | Normal(mu=0, sigma=1) |

## Examples

```python
import pyfcirt as fc

declaration = fc.GPCM()
example = fc.load_example("simulated_mgpcm", model=declaration, n_persons=3)
prepared = fc.PreparedInputs.from_inputs(example.data)
print(fc.data_coverage(prepared).to_string())
```

```text
                       n_items  size  n_persons  n_answered  n_missing
model_response_format
GPCM                        12     1          3          36          0
```

## See also

[Equations and identification](../../models/single_items.md#gpcm) and
[worked analysis](../../tutorials/binary_ordinal_workflow.ipynb).
