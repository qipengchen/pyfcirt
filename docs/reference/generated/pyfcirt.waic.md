# waic

```{eval-rst}
.. autofunction:: pyfcirt.waic
```

Compute the widely applicable information criterion from pointwise log likelihood.

## Parameters

`log_likelihood` : xarray.DataArray
: Finite MCMC response log likelihood with nonempty labelled `chain`,
  `draw` and `obs` dimensions, as returned by `fc.log_likelihood()`.

## Returns

xarray.Dataset
: `elpd_waic` is on the log predictive-density scale;
  `WAIC = -2 * elpd_waic` is on the deviance scale. The `obs` axis
  labels `elpd_waic_i`, `p_waic_i` and `waic_i`. Totals include
  `p_waic`, standard errors, sample and observation counts, and `warning`.

## Notes

WAIC uses the pointwise posterior log-likelihood variance for its penalty. A warning is raised when any pointwise variance exceeds 0.4, which occurs in this example. Compare models only on the same observations and observation unit.

## Examples

Use a fitted `Model`. The output below comes from the [HEXACO TIRT example](../../tutorials/tirt_workflow.ipynb).
Use the full posterior for model comparison. This example uses 32 draws per chain to show the output.

```python
import pyfcirt as fc

small = fc.Model(model.prepared)
small.attach_samples(model.idata.isel(draw=slice(0, 32)))
logs = small.log_likelihood(store=False, progressbar=False)
result = fc.waic(logs)
print(result[["elpd_waic", "p_waic", "WAIC"]].to_array().to_series().round(2).to_string())
```

```text
variable
elpd_waic   -4295.70
p_waic        615.83
WAIC         8591.39
```

## See also

[Prediction and model checks](../prediction.md).
