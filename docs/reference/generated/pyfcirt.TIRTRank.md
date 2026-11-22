# TIRTRank

```{eval-rst}
.. autoclass:: pyfcirt.TIRTRank
```

Joint Gaussian probabilities for full and partial ranking events.

## Parameters

`priors` : mapping or None
: Override local prior roles with unnamed PyMC distributions such as
  `pm.Normal.dist(mu=0, sigma=1)`. `None` uses the defaults below.

`loading_anchors` : mapping or None
: Statement IDs and positive fixed loadings. The default uses the design's anchors.
  Add references if preparation flags a missing anchor.

`reference_statements` : tuple
: IDs of statements with zero intercepts. `()` lets preparation choose references.

## Returns

TIRTRank
: A measurement declaration for `FCInput(..., model=declaration)`. Pass the configured
  declaration as the `model` value when loading a bundled dataset.

## Notes

Declare `"rank"`, `"pick"` or `"mole"` in the design. Fitting uses a Gaussian
 contrast constrained to the observed event. Prediction and pointwise log
 likelihood use its marginal event probability.

Default priors for active roles:

| Prior role | Default law |
| --- | --- |
| `loading_magnitude` | LogNormal(mu=0, sigma=0.5) |
| `ranking_intercept` | Normal(mu=0, sigma=1) |
| `residual_variance` | HalfNormal(loc=0, scale=1) |

## Examples

```python
import pyfcirt as fc

declaration = fc.TIRTRank()
example = fc.load_example("hexaco_fc3", model=declaration, n_persons=3)
model = fc.Model()
model.add_fc(
    example.data.design,
    example.data.responses.frame,
    format="block_long",
    response_format="mole",
    model=declaration,
)
model.prepare()
print(model.data_coverage().to_string())
```

```text
                       n_blocks  size  n_persons  n_answered  n_missing
model_response_format
TIRTRank / mole              20     3          3          60          0
```

## See also

[Equations and identification](../../models/tirt_rank.md) and
[worked analysis](../../tutorials/tirt_rank_workflow.ipynb).
