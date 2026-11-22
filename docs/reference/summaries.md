# Summaries and plotting

Inspect the specification with `Model.summary()` before fitting. Use `coef()`
for parameter estimates and `trait_summary()` for population moments. Use the
data coverage summary to check recorded and missing administrations.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~pyfcirt.Model.summary
   ~pyfcirt.coef
   ~pyfcirt.trait_summary

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
   ~pyfcirt.plot_ridge
```
