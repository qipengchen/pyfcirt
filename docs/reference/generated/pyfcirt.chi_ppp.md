# chi_ppp

```{eval-rst}
.. autofunction:: pyfcirt.chi_ppp
```

Compare observed and replicated Pearson discrepancies at matched posterior states.

## Parameters

`model` : Model
: Fitted MCMC model with observed categorical responses.

`predictions` : xarray.DataTree or None
: Existing replicas from this fit. If omitted and `draws=None`, stored
  replicas are reused when available.

`blocks`, `items` : sequence or None
: Limit the check by block or item ID.

`groupby` : str
: `"overall"` (default), `"person"`, `"block"`, `"item"`,
  `"measurement"` or `"pair"`. Use pair grouping for direct pair factors.

`draws` : int or None
: Number of states for fresh replicas. Cannot be supplied with `predictions`.

`random_seed` : int or None
: Seed for fresh predictions.

`save_discrepancy` : bool
: When storing results, also save per-observation log discrepancies.
  Default: `False`.

`store` : bool
: Save results on the model. Default: `True`.

`progressbar` : bool
: Display progress. Default: `True`.

## Returns

xarray.Dataset
: `PPP` is the posterior predictive probability of the replicated
  discrepancy being at least as large as the observed one.
  `PPP_mcse` estimates Monte Carlo uncertainty for ordered states when available. `n_obs` counts factors.
  `D_obs`, `D_rep` and `D_diff` keep the `chain` and `draw` axes
  and any chosen grouping dimension.

## Notes

Use stored replicas only with their original fit. Continuous Beta ratings need a continuous predictive check. Extreme PPP values suggest the selected discrepancy exposes a mismatch.

## Examples

Use a fitted `Model`. The output below comes from the [HEXACO TIRT example](../../tutorials/tirt_workflow.ipynb).
Check the three pairs in B1 separately. Each row reports the predictive check and the number of observed responses for that pair.

```python
check = model.chi_ppp(
    blocks=["B1"],
    groupby="pair",
    draws=64,
    random_seed=45,
    store=False,
    progressbar=False,
)
columns = ["statement_1", "statement_2", "n_obs", "PPP"]
print(check[["PPP", "n_obs"]].to_dataframe()[columns].round(2).to_string(index=False))
```

```text
statement_1 statement_2  n_obs  PPP
         i1          i2    120 0.53
         i1          i3    120 0.22
         i2          i3    120 0.27
```

## See also

[Prediction and model checks](../prediction.md).
