# Saving and continuing an analysis

Install `pyfcirt[io]` for the NetCDF backend.

## Calibration fits

```python
from pathlib import Path
import pyfcirt as fc

fitted.save(Path("results") / "calibration", include_log_likelihood=True)
saved = fc.load(Path("results") / "calibration")
posterior = saved.idata["posterior"]
fitted_again = saved.rebuild()
replicated = fitted_again.posterior_predictive(random_seed=42)
scores = fitted_again.score(new_data, random_seed=42)
```

Loading restores arrays and metadata without constructing or executing a model.
`SavedFit.rebuild()` explicitly reconstructs a supported model from saved data and
resolved priors, checks compatibility and returns a normal `FitResult`. It does
not refit calibration parameters. Original response encoding, missingness, labels,
design tables and parameter roles are retained. Integer and string labels remain
distinct.

## Scoring results

```python
scores.save(Path("results") / "new_people")
restored = fc.load(Path("results") / "new_people")
restored.summary()
restored.diagnostics()
```

A scoring bundle restores a `ScoreResult` with one native posterior, its fixed
calibration point, person status and provenance. No rebuild is needed for
summaries or diagnostics. Scoring metadata version 2 rejects earlier score
bundles using the removed conditional-mixture representation; calibration-fit
bundles are unchanged.

## Bundle format and boundaries

Each new bundle contains `metadata.json`, `posterior.nc` and `manifest.json`.
The manifest records format version, result type and file checksums, and is
written last. Existing destinations are not overwritten. JSON contains data and
settings; NetCDF contains a native xarray DataTree. Typed coordinate codecs
preserve mixed identifiers when NetCDF alone cannot.

The format does not use pickle or execute saved callbacks. Arbitrary native PyMC
graph edits are not portable through this format. Saving checks the declared
coordinates, density graph, deterministic expressions and shared dimension values
against a fresh reconstruction. Changed priors, observations and extra graph
terms are rejected, as are non-JSON settings. The check requires structural
agreement; mathematically equivalent custom rewrites may also be rejected.
Preserve source and environment versions for
custom models. Model-version checks protect supported rebuilds but do not promise
compatibility with every future PyMC or PyFCIRT version.


The centered MGPPM graph uses statistical model version 2. Earlier development
results with the noncentered version 1 remain readable as saved draws, but
explicit rebuilding rejects that older graph version. Refit with the current
centered model rather than silently reinterpreting sampling coordinates.
