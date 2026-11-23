# Population and covariates

Use `model.set_population()` before `model.prepare()` to define groups
and trait scales, or pass `PopulationSpec` directly to `model.prepare()`. Use
`Model.add_regressions()` to add predictors of abilities or eligible
measurement parameters. Feature tables use the entity IDs in the prepared design.
After preparation, call `model.available_covariate_targets()` to see the
selectors and feature tables for that design. For example, inspect the rows
for comparison cutpoints before adding a pair-level predictor:

```python
model.available_covariate_targets(level="pair", kind="regression")
```

For groups with distinct means on a common residual scale, set
`theta_correlation="shared"` and set `theta_residual_scale="unit"`.

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


## Register a grouped population

```{eval-rst}
.. automethod:: pyfcirt.Model.set_population
```

Call before preparation. Supply a person table keyed by `person_id`.
Repeated calls replace the pending declaration. With a group column, the
default estimates means and residual scales outside the reference group,
and estimates one trait correlation matrix per group.
