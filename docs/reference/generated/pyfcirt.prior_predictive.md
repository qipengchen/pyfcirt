# prior_predictive

```{eval-rst}
.. autofunction:: pyfcirt.prior_predictive
```

Draw parameter values and responses from the model's priors.

## Parameters

`model` : Model
: Prepared model, which may be unfitted.

`draws` : int
: Positive number of prior draws. Default: 500.

`random_seed` : int or None
: Reproducible seed.

`save_responses` : bool
: Also keep replicated answers on the model when storing. Default: `False`.

`store` : bool
: Store requested groups in `model.idata`. Default: `True`.

## Returns

xarray.DataTree
: `prior` has parameter draws and `prior_predictive` has simulated answers
  on `chain`, `draw` and batch-local response axes. Observed answers and
  design metadata are included when available. Replica draws are returned even
  when `save_responses=False`.

## Examples

Inspect the dimensions of eight prior replicates for one item and three respondents.

```python
import pyfcirt as fc

bundle = fc.load_example("simulated_mirt", model="DichotomousIRT", n_persons=3)
model = fc.Model(bundle.data)
prior = fc.prior_predictive(model, draws=8, random_seed=42, store=False)
view = fc.predictive_view(prior, item="I1")
print(dict(view.prior_predictive.response.sizes))
```

```text
{'chain': 1, 'draw': 8, 'obs': 3}
```

## See also

[Prediction and model checks](../prediction.md).
