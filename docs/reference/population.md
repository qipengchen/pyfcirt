# Population and covariates

Use `PopulationSpec` to define groups and trait scales. Use
`Model.add_regressions()` to add predictors of abilities or eligible
measurement parameters. Feature tables use the entity IDs in the prepared design.
For groups with distinct means on a common residual scale, set
`theta_correlation="shared"` and keep `theta_residual_scale="unit"`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   ~pyfcirt.PopulationSpec
   ~pyfcirt.Regression
   ~pyfcirt.RegressionOptions
   ~pyfcirt.Model.add_regressions
```
