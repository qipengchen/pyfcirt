# Supported scope and assumptions

PyFCIRT supplies six forced-choice declarations and three single-item declarations,
native PyMC measurement distributions, JAX probability kernels and labelled xarray
state. See [API](api.md) and [models](models/index.md).

Declare complete design, original response format and possible categories independently of
the data-table format. Preserve typed labels, statement directions and missingness.
Inspect identification, priors, convergence and posterior predictive fit before
interpreting estimates. A completed fit is not validation.

TIRT/GTUM comparison factors and joint ranking events have different observation
units. Compare LOO/WAIC/DIC only for matching data and units. Independent scoring
conditions on one fixed calibration point. Use `score_model().build()` with native
PyMC for other estimators and supply final-scale uncertainty yourself.
Non-MCMC package analysis is
limited to explicit information and reliability routes.

Custom graph expressions require caller-owned reconstruction and native persistence.
