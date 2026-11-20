# New-respondent scoring

Prepare a new person's responses with the calibrated design and response
coding. `score_model()` prepares a separate model. `score()` also fits it.

## Choose the calibration law

- **Fixed calibration point** is the default. Provide final parameters,
  use an existing fixed snapshot, or condition on the fitted posterior means.
- **Posterior calibration (experimental)** carries a distribution for selected shared
  parameters. The first round pools all supported free shared parameters
  into one group by default. Later rounds reuse that selection and grouping.
  The `auto` method chooses a density family each round. The new respondents
  get their own ability scores and must be outside the earlier evidence.

Supply any group labels or covariates required by the fitted reference.
Supply mixed responses as a mapping with `"fc"` and/or `"items"` keys and
one `ResponseTable` per supplied channel.

## Retain calibration uncertainty

Posterior-approximation scoring and recursive calibration are experimental.
Install PyFCIRT with the `posterior` extra to fit these approximations.

Calibrate with `Model.fit(retain_calibration=True)`, then call
`score_model(..., calibration="posterior")` or `score(..., calibration="posterior")`.
To update a subset, pass an explicit `approximation` from
`Model.posterior_approximation(var_names=...)`.
See [the recursive scoring example](../tutorials/recursive_scoring.ipynb)
for how one update supplies the prior for the next batch.

```{toctree}
:hidden:
generated/pyfcirt.score
generated/pyfcirt.score_model
```
