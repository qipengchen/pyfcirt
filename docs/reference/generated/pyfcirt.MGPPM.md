# MGPPM

```{eval-rst}
.. autoclass:: pyfcirt.MGPPM
```

Dominance model with probabilities normalized over ranking patterns.

## Parameters

`priors` : mapping or None
: Override local prior roles with unnamed PyMC distributions such as
  `pm.Normal.dist(mu=0, sigma=1)`. `None` uses the defaults below.

`reference_statements` : tuple
: Statement IDs to use as zero-intercept references. Default: `()` for
  preparation's choice.

## Returns

MGPPM
: A measurement declaration for `FCInput(..., model=declaration)`. Pass the configured
  declaration as the `model` value when loading a bundled dataset.

## Notes

Supports RANK, PICK and MOLE and graded comparisons in dyads. Use
 `graded_intercept` for the graded pair extension and `ranking_intercept`
 for ranking tasks.

Default priors for active roles:

| Prior role | Default law |
| --- | --- |
| `loading_magnitude` | LogNormal(mu=0, sigma=0.5) |
| `ranking_intercept` | Normal(mu=0, sigma=2) |
| `graded_intercept` | Normal(mu=0, sigma=2) |

## Examples

```python
import pyfcirt as fc

declaration = fc.MGPPM()
example = fc.load_example("simulated_mgppm", model=declaration, n_persons=3)
prepared = fc.PreparedInputs.from_inputs(example.data)
print(fc.data_summary(prepared).to_string())
```

```text
                       n_blocks  size  n_persons  n_answered  n_missing
model_response_format
MGPPM / rank                 24     3          3          72          0
```

## See also

[Equations and identification](../../models/mgppm.md) and
[worked analysis](../../tutorials/mgppm_workflow.ipynb).
