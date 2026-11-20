# plot_ridge

```{eval-rst}
.. autofunction:: pyfcirt.plot_ridge
```

Plot marginal posterior distributions with credible intervals.

## Parameters

`data` : xarray.Dataset or xarray.DataTree
: Posterior with the selected variable and at least two sampled states on
  the `chain` and `draw` axes.

`var_name` : str
: Posterior variable. Default: `"theta"`.

`coords` : mapping or None
: Select parameter labels such as `person` or `trait`.

`max_rows` : int or None
: Row limit after selection. Default: 12. `None` includes all rows.

`col_wrap` : int
: Maximum panel columns. Default: 4.

`ci_probs` : tuple of float
: Credible interval masses. Default: `(0.5, 0.95)`.

`figsize` : tuple or None
: Figure size in inches.

`colors` : str, sequence or None
: Matplotlib color or palette. `None` uses the default color.

## Returns

arviz_plots.PlotCollection
: Editable ridge plot. People, statements or items form rows when
  available. Other parameter dimensions remain facets.

## Examples

Use a fitted `Model`. The output below comes from the [HEXACO TIRT example](../../tutorials/tirt_workflow.ipynb).
Compare Emotionality estimates for three respondents.

```python
import pyfcirt as fc

people = model.idata.posterior.person.values[:3].tolist()
plot = fc.plot_ridge(
    model.idata,
    coords={"person": people, "trait": ["emotionality"]},
    figsize=(5, 2.8),
)
```

```{image} ../../_static/reference/ridge.png
:alt: Emotionality posterior distributions for three respondents, with 50% and 95% intervals.
```

## See also

[Summaries and plotting](../summaries.md).
