# residuals

```{eval-rst}
.. autofunction:: pyfcirt.residuals
```

Compute observed-minus-expected responses at each posterior state.

## Parameters

`model` : Model
: Fitted MCMC model with observed responses.

`progressbar` : bool
: Display forward evaluation progress. Default: `True`.

## Returns

xarray.Dataset
: One residual variable per batch. The array uses `chain`, `draw` and
  that batch's observation axis. Unordered responses add a category axis
  for one-hot observed indicators minus predicted probabilities. Numeric
  categories and Beta scores use observed minus conditional mean.

## Notes

These are unstandardized residuals.

## Examples

Use a fitted `Model`. The output below comes from the [HEXACO TIRT example](../../tutorials/tirt_workflow.ipynb).
The example uses 32 draws per chain for a small preview and inspects the first residual array.

```python
import pyfcirt as fc

small = fc.Model(model.prepared)
small.attach_samples(model.idata.isel(draw=slice(0, 32)))
errors = small.residuals(progressbar=False)
first = next(iter(errors.data_vars))
print(dict(errors[first].sizes))
```

```text
{'b0_obs': 360, 'chain': 4, 'draw': 32}
```

## See also

[Prediction and model checks](../prediction.md).
