# trait_summary

```{eval-rst}
.. autofunction:: pyfcirt.trait_summary
```

Summarize the population's trait means, SDs and correlations.

## Parameters

`data` : Model
: Prepared model and the people whose groups and predictors define the population.

`idata` : xarray.DataTree or None
: Compatible posterior. `None` uses model samples.

`by_group` : bool
: Separate moments for each actual population group. Default: `False`.

`ci_prob` : float
: Credible interval mass. Default: 0.95.

`ci_kind` : {"hdi", "eti"}
: Highest density (default) or equal-tailed interval.

`summary_kwargs`
: ArviZ summary options such as `coords`, `var_names`, `kind` and
  `round_to`. If selecting sample states, use the original `chain` and
  `draw` labels.

## Returns

xarray.Dataset
: `population_mean` and `population_sd` have `trait` and `summary`
  dimensions. `trait_correlation` adds `trait_aux`. With groups selected,
  `population_group` labels the groups. Fixed moments have missing uncertainty.

## Notes

With predictors, marginal correlations incorporate differences in predicted means across the reference people as well as the conditional residual variation.

## Examples

Use a fitted `Model`. The output below comes from the [HEXACO TIRT example](../../tutorials/tirt_workflow.ipynb).
Select correlations between Honesty–Humility and two other traits.

```python
moments = model.trait_summary(kind="stats")
row = moments["trait_correlation"].sel(
    trait="honestyhumility",
    trait_aux=["agreeableness", "emotionality"],
    summary=["mean", "hdi95_lb", "hdi95_ub"],
)
print(row.to_pandas().round(2).to_string())
```

```text
summary        mean  hdi95_lb  hdi95_ub
trait_aux
agreeableness  0.35      0.06      0.63
emotionality  -0.01     -0.32      0.29
```

## See also

[Summaries and plotting](../summaries.md).
