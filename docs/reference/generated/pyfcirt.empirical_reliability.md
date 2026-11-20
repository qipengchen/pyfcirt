# empirical_reliability

```{eval-rst}
.. autofunction:: pyfcirt.empirical_reliability
```

Compute reliability from explicit score means and variances.

## Parameters

`mean` : xarray.DataArray
: Finite final score means labelled by `person` and optionally `trait`.

`variance` : xarray.DataArray
: Nonnegative final score variances with the same labels and dimensions.
  Use at least two respondents.

`method` : {"ratio", "subtraction"}
: For score sample variance V and mean posterior variance U, use
  `V/(V+U)` (default) or `1-U/V`.

## Returns

xarray.Dataset
: `reliability`, `score_variance` and `mean_posterior_variance` by
  trait when that dimension is present; `n_person` is a scalar count.

## Examples

Compute reliability for neuroticism from the TIRT posterior moments. Use the same call with labelled means and variances from another estimator.

```python
import pyfcirt as fc

theta = model.idata.posterior["theta"].sel(trait="n")
mean = theta.mean(("chain", "draw"))
variance = theta.var(("chain", "draw"), ddof=1)
result = fc.empirical_reliability(mean, variance)
print(round(result.reliability.item(), 2))
```

```text
0.77
```

## See also

[Information and reliability](../information.md).
