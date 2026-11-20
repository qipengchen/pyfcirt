# coef

```{eval-rst}
.. autofunction:: pyfcirt.coef
```

Summarize measurement or regression parameters from posterior samples.

## Parameters

`data` : Model
: Prepared model identifying parameters and units. `model.coef()` binds it.

`idata` : xarray.DataTree or None
: Compatible posterior. `None` uses the model's current samples.

`component` : {"measurement", "regression"}
: Measurement parameters (default), or regression terms and scales.

`blocks`, `items` : sequence or None
: Ordered block or item IDs. Omit both to include all measurements. A selection
  in just one channel omits the other. Empty sequences select no units.
  Use these arguments only for measurement summaries.

`ci_prob` : float
: Credible interval mass. Default: 0.95.

`ci_kind` : {"hdi", "eti"}
: Highest density (default) or equal-tailed interval. Median summaries use ETIs.

`var_names` : str, sequence of str or None
: Canonical names such as `"loading"` and `"cutpoint"`. The default omits
  TIRT statement effects; request `"statement_effect"` to include them.

`coords` : mapping or None
: Select dimension labels or applicable auxiliary coordinate values.
  Use `blocks` and `items` for original design IDs.

`kind` : str
: ArviZ summary mode. Default `"all"` includes diagnostics. Use `"stats"`
  for just estimates and intervals.

`summary_kwargs`
: Additional ArviZ options such as `round_to` and `filter_vars`.

## Returns

pandas.DataFrame
: The measurement index identifies each unit and parameter, with applicable
  statement, trait, pair or category labels. A mixed summary includes channel.
  Regression rows identify target and term, and any applicable traits or groups.
  Fixed rows show their values under `mean` and leave uncertainties blank.

## Notes

The reported `loading` precedes design keying. A pair labelled `A > B`
uses high responses to favor A. The actual answer is in `response_data()`.

## Examples

Use a fitted `Model`. The output below comes from the [HEXACO TIRT example](../../tutorials/tirt_workflow.ipynb).
In block B1, loadings use statement labels and cutpoints use pair labels.

```python
table = model.coef(blocks=["B1"], kind="stats", round_to=2).reset_index()
table["label"] = table["statement"].fillna(table["pair"])
columns = ["label", "parameter", "mean", "sd", "hdi95_lb", "hdi95_ub"]
print(table[columns].to_string(index=False, na_rep=""))
```

```text
  label         parameter  mean   sd  hdi95_lb  hdi95_ub
     i1           loading  1.46 0.37      0.79      2.18
     i2           loading  0.60 0.18      0.27      0.98
     i3           loading  0.55 0.18      0.21      0.89
     i1 residual_variance  0.50
     i2 residual_variance  1.13 0.59      0.02      2.22
     i3 residual_variance  1.37 0.62      0.31      2.68
i1 > i2          cutpoint  0.45 0.70     -1.00      1.74
i1 > i3          cutpoint  0.30 0.69     -1.02      1.64
i2 > i3          cutpoint  0.25 0.74     -1.16      1.73
```

## See also

[Summaries and plotting](../summaries.md).
