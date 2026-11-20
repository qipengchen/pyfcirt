# log_likelihood

```{eval-rst}
.. autofunction:: pyfcirt.log_likelihood
```

Evaluate the response log likelihood of each observed factor.

## Parameters

`model` : Model
: Fitted MCMC model with actual observed response factors.

`store` : bool
: Save the pooled array in `model.idata.log_likelihood`. Default: `True`.

`progressbar` : bool
: Display progress for missing factors. Default: `True`.

## Returns

xarray.DataArray
: `response(chain, draw, obs)`. Each `obs` identifies an actual likelihood
  factor and carries prepared person, measurement and, where applicable, pair
  indices. Existing matching factors are reused.

## Notes

Likelihood is conditional on the fitted parameters and person traits. TIRTRank returns the observed event's marginal log probability, while ordinary pair models return their comparison factors. Compare criteria only for the same observations and observation unit.

## Examples

Use a fitted `Model`. The output below comes from the [HEXACO TIRT example](../../tutorials/tirt_workflow.ipynb).
The example uses 32 draws per chain for a small preview. The obs axis identifies individual comparisons in this TIRT fit.

```python
import pyfcirt as fc

small = fc.Model(model.prepared)
small.attach_samples(model.idata.isel(draw=slice(0, 32)))
logs = small.log_likelihood(store=False, progressbar=False)
print(dict(logs.sizes))
```

```text
{'chain': 4, 'draw': 32, 'obs': 7200}
```

## See also

[Prediction and model checks](../prediction.md).
