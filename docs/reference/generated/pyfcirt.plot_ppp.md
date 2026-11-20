# plot_ppp

```{eval-rst}
.. autofunction:: pyfcirt.plot_ppp
```

Plot the posterior predictive discrepancy differences.

## Parameters

`check` : xarray.Dataset
: Returned Pearson check from `chi_ppp()`.

`coords` : mapping or None
: Select existing labels on the check's grouping axis.

`bins` : str, int or array-like
: Matplotlib histogram bins. Default: `"auto"`.

`quantile_range` : tuple or None
: Optional interval of display quantiles. `None` displays all draws.
  The interval does not change the reported PPP.

`max_panels` : int or None
: Panel limit. Default: 12. `None` uses every selected panel.

`figsize` : tuple or None
: Figure size in inches.

## Returns

matplotlib.figure.Figure
: Histograms of `D_rep - D_obs` with a reference line at zero and
  the computed PPP in each panel.

## Examples

Use a fitted `Model`. The output below comes from the [HEXACO TIRT example](../../tutorials/tirt_workflow.ipynb).
Compare the three pairs in B1.

```python
import pyfcirt as fc

check = model.chi_ppp(
    blocks=["B1"],
    groupby="pair",
    draws=64,
    random_seed=45,
    store=False,
    progressbar=False,
)
figure = fc.plot_ppp(check, quantile_range=(0.005, 0.995))
```

```{image} ../../_static/reference/ppp.png
:alt: Pearson discrepancy distributions for the three pairs in HEXACO block B1.
```

## See also

[Prediction and model checks](../prediction.md).
