# Likelihood and model comparison

After fitting, request conditional response log-likelihood explicitly:

```python
from pyfcirt.model_comparison import dic, waic

logp = fit.log_likelihood(progressbar=False)
dic_result = dic(logp)
waic_result = waic(logp)
loo_result = fit.loo(progressbar=False)
```

`logp` is an xarray DataArray with `chain`, `draw` and `obs_id` dimensions.
Each observation is one effective person-block response. Full ranking uses the
probability of the full order; PICK and MOLE use the sum over compatible orders;
graded uses its declared category. Missing and unadministered events are excluded.
Log-likelihood conditions on both person traits and item parameters, includes
category normalization, and excludes parameter priors and Jacobians. Its attrs
record this target, observation unit and data fingerprint.

This is a conditional person-block predictive target. It does not integrate out
person traits or describe prediction for an entirely new person. Compare models
only on the same responses, effective observations and predictive target; ranking
and graded values from different datasets cannot be ranked against one another.

## Outputs and scales

`dic(logp)` returns a Dataset with overall `Dbar`, `pV` and `DIC`, plus the three
chainwise components. It follows R2jags' default pV correction: for each chain,
compute the sample variance (`ddof=1`) of **total** response deviance, divide by
two and add mean deviance; then average the chainwise results. At least two draws
per chain are required. This differs from a plug-in mean-parameter DIC and from
a pooled variance that includes between-chain mean differences.

`waic(logp)` returns `elpd_waic`, `p_waic`, deviance-scale `WAIC`, their pointwise
terms and standard errors. The implementation matches ArviZ 0.22.0: posterior
variance and across-observation variance both use `ddof=0`. A log predictive
variance above 0.4 emits a reliability warning and sets `warning=True`. The old
pointwise-equals-total warning is also retained. Nonfinite values, duplicate
observation IDs and malformed sample dimensions are rejected rather than dropped.

Larger elpd and smaller deviance-scale WAIC/DIC indicate better relative fit under
their stated definitions. Assess sampling diagnostics and warnings before using
these values. Numerical agreement with a reference does not establish convergence.

`fit.loo()` returns the current native ArviZ PSIS-LOO result, including pointwise
elpd and Pareto-k by default. ArviZ calculates relative efficiency from posterior
draws unless `reff` is explicitly supplied; use `reff=1` only when justified.
`fit.dic()` and `fit.waic()` are convenience methods that compute log-likelihood
on request. For several metrics, retain `logp` and reuse the standalone functions.

To reuse the same computed log-likelihood with ArviZ directly:

```python
import arviz as az
import xarray as xr

comparison_data = xr.DataTree.from_dict(
    {
        "posterior": fit.idata["posterior"].to_dataset(),
        "log_likelihood": logp.to_dataset(),
    }
)
loo_result = az.loo(comparison_data, var_name="response", pointwise=True)
```

The fitted DataTree and its native graph remain unchanged. PyMC computes the
response logp; a detached graph normalizes untransformed value names to handle
the Cholesky transform naming issue in the current PyMC implementation. Tests
compare each event with independent scalar probabilities and confirm the original
graph's value names are unchanged.

## Independent numerical references

Ordinary tests require neither R nor legacy ArviZ. Optional reference tests run
separate executables against identical fixed arrays, including extreme values,
different chain means, constant draws and single-observation boundaries. The
comparison tolerance is `rtol=1e-10, atol=1e-10` for the float64 fixtures.

The DIC oracle executes the **unmodified original** `as.bugs.array2` conversion
function from R2jags 0.9-0, pinned to commit
`1caf7ac974ea45b1411e4cf36057ba026ca6db86`, with the original helper files and
R2WinBUGS' `monitor`. The test verifies normalized source hashes before executing
them. This loads the post-processing source rather than installing the full
R2jags/JAGS stack; no JAGS model or substitute DIC formula is run. The sources and
R2WinBUGS library stay outside the package. See the
[pinned upstream function](https://github.com/suyusung/R2jags/blob/1caf7ac974ea45b1411e4cf36057ba026ca6db86/R/as.bugs.array2.R).

The WAIC oracle calls the actual `arviz.waic` in an isolated `arviz==0.22.0`
environment, checking both log and deviance scales, pointwise terms, penalty,
standard error and warnings. No reference package is imported into the current
PyFCIRT process. See the
[legacy implementation](https://github.com/arviz-devs/arviz/blob/v0.22.0/arviz/stats/stats.py).

Configure these environment variables before selecting `pytest -m reference`:

- `PYFCIRT_LEGACY_ARVIZ_PYTHON`: isolated ArviZ 0.22.0 Python executable.
- `PYFCIRT_RSCRIPT`: Rscript executable.
- `PYFCIRT_R2JAGS_SOURCE`: upstream checkout matching the source pin.
- `PYFCIRT_R_LIBRARY`: optional isolated R library containing R2WinBUGS and coda.
- `PYFCIRT_REFERENCE_FITS`: optional directory of saved small ranking/graded fits.

Unconfigured reference checks are skipped explicitly. Configured-but-broken
environments, changed source files, timeouts and numerical mismatches fail.
The full reference gate is required at initial implementation, after metric
changes and before release. Saved real-fit checks compare the complete input
chain and a native NetCDF roundtrip without starting another sampler. This does
not replace the package-level save/rebuild workflow, which is a separate module.
