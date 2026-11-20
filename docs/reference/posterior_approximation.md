# Runtime posterior approximation

Fit an incoming calibration prior from saved posterior draws and use it to
score a new batch. Posterior-approximation scoring and recursive calibration
are experimental. `Model.posterior_approximation()` defaults to one joint
density across the automatically selected shared parameters and uses
`method="auto"` to select its family. Use `joint=False` on the initial
calibration to fit a separate density for each parameter kind.

## Statistical target

Posterior scoring combines the approximate calibration prior with the new
likelihood. The first round selects all supported free shared variables
by default, and later rounds reuse that selection and grouping. Shared variables outside
an explicit selection are fixed at their calibration values. New-person
traits and response auxiliaries retain their native laws. Previously included respondents are rejected before fitting.
If an update changes a 4PL lower asymptote, its dependent upper asymptote
must also be allowed to vary.

## Approximation methods

| Method | Distribution for each group |
| --- | --- |
| `mvn` | One full-covariance Gaussian |
| `mean_field` | Independent Gaussian coordinates |
| `gmm` | Full-covariance Gaussian mixture, requiring eligible held-out chains |
| `auto` (Model default) | Gaussian mixture when it clears the held-out threshold; otherwise one full-covariance Gaussian |

Use `components` to supply candidate mixture sizes. For `auto`, one
held-out chain selects the candidate and another assesses the gain.
The mixture must improve mean held-out log density by more than both
0.01 and twice the block standard error. Both densities are evaluated
against the same held-out posterior draws. The chosen density is refitted
on all available draws. The Model default repeats the audit on each round.
Fitting requires `pyfcirt[posterior]`.

## Scoring and saving

Use [Model.posterior_approximation()](generated/pyfcirt.Model.posterior_approximation.md)
to fit a prior that carries the PyFCIRT design and evidence. Keep the native
calibration RVs needed for the next update when saving samples. Because the
fitted prior is a runtime object, save the posterior together with
`fixed_parameters` and `calibration_evidence`, then restore compatible
samples into a matching model for analysis or to fit the next approximation.

Rebuilding a previous scoring graph for estimation or prior prediction
requires that round's original incoming approximation. For an example
with five successive batches, see [recursive scoring](../tutorials/recursive_scoring.ipynb).

```{toctree}
:hidden:
generated/pyfcirt.PosteriorApproximation
```
