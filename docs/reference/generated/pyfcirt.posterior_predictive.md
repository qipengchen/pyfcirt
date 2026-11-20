# posterior_predictive

```{eval-rst}
.. autofunction:: pyfcirt.posterior_predictive
```

Generate responses at fitted posterior states.

## Parameters

`model` : Model
: Fitted model with MCMC draws and final parameters used by the response law.

`draws` : int or None
: Number of joint posterior states to use. `None` keeps all states in their
  original chains; an explicit count selects states into one predictive chain.

`random_seed` : int or None
: Seed for state selection and simulated responses.

`include_probabilities` : bool
: Also return probabilities or Beta parameters. Default: `True`.

`store` : bool
: Keep the returned groups in `model.idata`. Default: `True`.

`progressbar` : bool
: Display progress. Default: `True`.

## Returns

xarray.DataTree
: `posterior_predictive` contains replicated responses over `chain`,
  `draw` and native observation axes. `response_probabilities` adds
  one category axis per categorical batch; Beta batches provide parameters
  in `response_parameters`. Use `predictive_view()` to align one unit
  with its observed answers.

## Examples

Use a fitted `Model`. The output below comes from the [HEXACO TIRT example](../../tutorials/tirt_workflow.ipynb).
Generate 24 replicates, then check the draw and observation counts for block B1.

```python
import pyfcirt as fc

predictions = model.posterior_predictive(
    draws=24,
    random_seed=45,
    store=False,
    progressbar=False,
)
view = fc.predictive_view(predictions, block="B1")
print(dict(view.posterior_predictive.response.sizes))
```

```text
{'chain': 1, 'draw': 24, 'obs': 360}
```

## See also

[Prediction and model checks](../prediction.md).
