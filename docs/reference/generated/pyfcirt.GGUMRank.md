# GGUMRank

```{eval-rst}
.. autoclass:: pyfcirt.GGUMRank
```

Sequential choice with ideal-point statement endorsement.

## Parameters

`priors` : mapping or None
: Override local prior roles with unnamed PyMC distributions such as
  `pm.Normal.dist(mu=0, sigma=1)`. `None` uses the defaults below.

`reference_statements` : tuple
: Statement IDs to use as zero-threshold references. Default: `()` for
  preparation's choice.

## Returns

GGUMRank
: A measurement declaration for `FCInput(..., model=declaration)`. Pass the configured
  declaration as the `model` value when loading a bundled dataset.

## Notes

Supports RANK, PICK and MOLE. In the design supply a `location_side` per
 statement: +1 for a positive ideal point, −1 for negative and 0 for
 unrestricted. A custom location prior replaces the corresponding sign-specific
 default.

Default priors for active roles:

| Prior role | Default law |
| --- | --- |
| `loading_magnitude` | LogNormal(mu=0, sigma=0.25) |
| `location_positive` | TruncatedNormal(mu=2, sigma=1, lower=0, upper=inf) |
| `location_negative` | TruncatedNormal(mu=-2, sigma=1, lower=-inf, upper=0) |
| `location_free` | Normal(mu=0, sigma=1) |
| `threshold` | Normal(mu=0, sigma=2) |

## Examples

```python
import pyfcirt as fc

declaration = fc.GGUMRank()
example = fc.load_example("simulated_ggum_rank", model=declaration, n_persons=3)
prepared = fc.PreparedInputs.from_inputs(example.data)
print(fc.data_summary(prepared).to_string())
```

```text
                       n_blocks  size  n_persons  n_answered  n_missing
model_response_format
GGUMRank / rank              24     3          3          72          0
```

## See also

[Equations and identification](../../models/ggum_rank.md) and
[worked analysis](../../tutorials/ggum_rank_workflow.ipynb).
