# GTUM

```{eval-rst}
.. autoclass:: pyfcirt.GTUM
```

Ideal-point pair comparisons with utility determined by distance.

## Parameters

`priors` : mapping or None
: Override local prior roles with unnamed PyMC distributions such as
  `pm.Normal.dist(mu=0, sigma=1)`. `None` uses the defaults below.

`link` : {"logit", "beta_logit"}
: Ordinary unscaled logit (default), or Beta with a logit mean. Beta needs a
  `UnitIntervalScale` and an explicit positive `response_precision` prior.

`block_effect` : bool
: Include a respondent-by-block effect when a block has more than two statements.
  Default: `False`. The Beta link does not support this effect.

## Returns

GTUM
: A measurement declaration for `FCInput(..., model=declaration)`. Pass the configured
  declaration as the `model` value when loading a bundled dataset.

## Notes

Use the design's `location_side` to distinguish positive, negative and
 unrestricted locations. Binary or graded pair responses have separate
 comparison factors; pairs derived from ranks use a composite likelihood.

Default priors for active roles:

| Prior role | Default law |
| --- | --- |
| `loading_magnitude` | LogNormal(mu=0, sigma=0.5) |
| `location_positive` | TruncatedNormal(mu=2, sigma=1, lower=0, upper=inf) |
| `location_negative` | TruncatedNormal(mu=-2, sigma=1, lower=-inf, upper=0) |
| `location_free` | Normal(mu=0, sigma=1) |
| `cutpoints` | Normal(mu=0, sigma=1) |
| `block_effect_scale` | HalfNormal(loc=0, scale=1) |
| `person_block_effect` | Normal(mu=0, sigma=1) |

## Examples

```python
import pyfcirt as fc

declaration = fc.GTUM()
example = fc.load_example("simulated_gtum", model=declaration, n_persons=3)
prepared = fc.PreparedInputs.from_inputs(example.data)
print(fc.data_summary(prepared).to_string())
```

```text
                       n_blocks  size  n_persons  n_answered  n_missing
model_response_format
GTUM                         24     3          3          72          0
```

## See also

[Equations and identification](../../models/gtum.md) and
[worked analysis](../../tutorials/gtum_workflow.ipynb).
