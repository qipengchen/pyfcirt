# TIRT

```{eval-rst}
.. autoclass:: pyfcirt.TIRT
```

Linear statement utilities for pair comparisons.

## Parameters

`priors` : mapping or None
: Override local prior roles with unnamed PyMC distributions such as
  `pm.Normal.dist(mu=0, sigma=1)`. `None` uses the defaults below.

`link` : {"probit", "logit", "beta_probit", "beta_logit"}
: Default: `"probit"`. The discrete logit multiplies the standardized predictor by
  1.702. Beta links require a separate pair judgment, a `UnitIntervalScale`
  and an explicit positive `response_precision` prior.

`residuals` : {"estimated", "fixed"}
: Estimate eligible statement residual variances (default), or fix them to 0.5.

`statement_effect_parameterization` : {"centered", "noncentered"}
: Parameterization of discrete statement effects. Default: `"noncentered"`.

`loading_anchors` : mapping or None
: Statement IDs and positive fixed loadings. `None` uses design and automatic references.

## Returns

TIRT
: A measurement declaration for `FCInput(..., model=declaration)`. Pass the configured
  declaration as the `model` value when loading a bundled dataset.

## Notes

Binary and graded pairs have one likelihood factor per comparison. For
 comparisons derived from a ranking, the product is a composite likelihood.
 Estimated statement effects are scaled by the statement residual standard
 deviation.

Default priors for active roles:

| Prior role | Default law |
| --- | --- |
| `loading_magnitude` | LogNormal(mu=0, sigma=0.5) |
| `cutpoints` | Normal(mu=0, sigma=1) |
| `residual_variance` | HalfNormal(loc=0, scale=1) |
| `statement_effect` | Normal(mu=0, sigma=1) |

## Examples

```python
import pyfcirt as fc

declaration = fc.TIRT(link="logit", residuals="fixed")
example = fc.load_example("tirt_study", model=declaration, n_persons=3)
prepared = fc.PreparedInputs.from_inputs(example.data)
print(fc.data_coverage(prepared).to_string())
```

```text
                       n_blocks  size  n_persons  n_answered  n_missing
model_response_format
TIRT                         24     3          3          72          0
```

## See also

[Equations and identification](../../models/tirt.md) and
[worked analysis](../../tutorials/tirt_workflow.ipynb).
