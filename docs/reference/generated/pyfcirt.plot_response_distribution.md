# plot_response_distribution

```{eval-rst}
.. autofunction:: pyfcirt.plot_response_distribution
```

Plot observed category proportions or counts.

## Parameters

`summary` : pandas.DataFrame
: Table from `response_summary()` or the long responses from `response_data()`.

`statistic` : {"proportion", "count"}
: Bar height. Default: `"proportion"`.

`kwargs`
: Seaborn `catplot()` options such as `height` and `col_wrap`.
  An optional `hue` names a grouping column. Category axes are local.

## Returns

seaborn.axisgrid.FacetGrid
: One panel per applicable unit, pair and group. Edit its axes with
  Seaborn or Matplotlib.

## Examples

Compare observed preferences within each pair of block B1.

```python
import pyfcirt as fc

bundle = fc.load_example("tirt_study", model="TIRT", n_persons=6)
model = fc.Model(bundle.data)

answers = model.response_data(blocks=["B1"])
counts = fc.response_summary(answers)
grid = fc.plot_response_distribution(counts, height=2.6)
```

```{image} ../../_static/reference/response_distribution.png
:alt: Observed category proportions for each of the three pairs of B1.
```

## See also

[Summaries and plotting](../summaries.md).
