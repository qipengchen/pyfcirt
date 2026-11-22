# Population and covariates

Use `PopulationSpec` to define groups and trait scales. Use
`Model.add_regressions()` to add predictors of abilities or eligible
measurement parameters. Feature tables use the entity IDs in the prepared design.
After preparation, call `model.available_covariate_targets()` to see the
selectors and feature tables for that design. For example, inspect the rows
for comparison cutpoints before adding a pair-level predictor:

```python
model.available_covariate_targets(level="pair", kind="regression")
```

For groups with distinct means on a common residual scale, set
`theta_correlation="shared"` and keep `theta_residual_scale="unit"`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~pyfcirt.PopulationSpec
   ~pyfcirt.Regression
   ~pyfcirt.RegressionOptions
   ~pyfcirt.Model.available_covariate_targets
   ~pyfcirt.Model.add_regressions
```
