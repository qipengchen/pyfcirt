# GGUM

```{eval-rst}
.. autoclass:: pyfcirt.GGUM
```

Generalized graded unfolding for ordered item responses.

## Parameters

`priors` : mapping or None
: Override local prior roles with unnamed PyMC distributions such as
  `pm.Normal.dist(mu=0, sigma=1)`. `None` uses the defaults below.


## Returns

GGUM
: A measurement declaration for `ItemInput(..., model=declaration)`. Pass the configured
  declaration as the `model` value when loading a bundled dataset.

## Notes

Specify a positive (+1), negative (−1) or unrestricted (0) location in
 each active item–trait row. The first category step is zero. Other steps
 can be negative or positive and are estimated without ordering constraints.

Default priors for active roles:

| Prior role | Default law |
| --- | --- |
| `loading_magnitude` | LogNormal(mu=0, sigma=0.5) |
| `location_positive` | TruncatedNormal(mu=1, sigma=1, lower=0, upper=inf) |
| `location_negative` | TruncatedNormal(mu=-1, sigma=1, lower=-inf, upper=0) |
| `location_free` | Normal(mu=0, sigma=1) |
| `threshold` | Normal(mu=0, sigma=1) |

## Examples

```python
import pyfcirt as fc

declaration = fc.GGUM()
example = fc.load_example("simulated_mggum", model=declaration, n_persons=3)
prepared = fc.PreparedInputs.from_inputs(example.data)
print(fc.data_summary(prepared).to_string())
```

```text
                       n_items  size  n_persons  n_answered  n_missing
model_response_format
GGUM                        12     1          3          36          0
```

## See also

[Equations and identification](../../models/single_items.md#ggum) and
[worked analysis](../../tutorials/binary_ordinal_workflow.ipynb).
