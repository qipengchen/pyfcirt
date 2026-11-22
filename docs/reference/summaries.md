# Summaries and plotting

Inspect the model specification with
[`model.summary()`](generated/pyfcirt.Model.summary.md) before fitting.

```{toctree}
:hidden:

generated/pyfcirt.Model.summary
```

## Posterior results

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~pyfcirt.coef
   ~pyfcirt.trait_summary
   ~pyfcirt.plot_ridge
```

## Observed response distributions

Use `model.response_data()` to extract the observed answers. Then pass those
rows to `fc.response_summary()` to count each answer, or plot the counts with
`fc.plot_response_distribution()`. These summaries use observed answers as
their denominators. For counts of administered, answered and missing units,
use [`model.data_summary()`](generated/pyfcirt.data_summary.md).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~pyfcirt.response_data
   ~pyfcirt.response_summary
   ~pyfcirt.plot_response_distribution
```
