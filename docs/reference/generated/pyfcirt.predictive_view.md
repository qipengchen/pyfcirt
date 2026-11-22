# predictive_view

```{eval-rst}
.. autofunction:: pyfcirt.predictive_view
```

Align the observed and replicated answers for a single unit.

## Parameters

`idata` : xarray.DataTree
: Tree from `posterior_predictive()` or `prior_predictive()`, with
  observed responses and its packaged design/support groups.

`block`, `item` : str, int or None
: Exactly one block ID or item ID. The selected unit must have one
  consistent response support.

## Returns

xarray.DataTree
: `observed_data.response(obs)` and `posterior_predictive.response(chain, draw, obs)`
  or its prior-predictive counterpart. They share the original `obs` IDs.
  `response_support` describes the selected unit's category values or patterns.

## Examples

Use a fitted `Model`. The output below comes from the [HEXACO TIRT example](../../tutorials/tirt_workflow.ipynb).
Select block B1 and verify that the observed and replicated responses share the same observation IDs.

```python
import pyfcirt as fc

predictions = model.posterior_predictive(
    draws=24,
    random_seed=45,
    store=False,
    progressbar=False,
)
view = fc.predictive_view(predictions, block="B1")
observed = view.observed_data["response"]
replicated = view.posterior_predictive["response"]
print(observed.get_index("obs").equals(replicated.get_index("obs")))
```

```text
True
```

## See also

[Prediction and model checks](../prediction.md).
