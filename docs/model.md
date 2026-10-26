# Model construction and prior prediction

The implemented model family is MGPPM. Start with a validated `FCData`, as
shown in the [data guide](data.md). Configuration, graph construction and forward
generation are separate operations:

```python
from pyfcirt import Model, Prior
from pyfcirt.models import MGPPM

model = Model(
    data,
    family=MGPPM(
        priors={
            "loading_magnitude": Prior("LogNormal", mu=0, sigma=0.5),
        }
    ),
    priors={"trait_correlation": Prior("LKJCorr", eta=2)},
)
print(model.prior_summary())
built = model.build()
prior = built.prior_predictive(draws=100, random_seed=42)
replicated = built.response_frame(prior, draw=0)
```

`prior` is the native PyMC `xarray.DataTree`, with `prior`, `prior_predictive`
and `observed_data` groups. `replicated` contains one simulated response for each
effective person-block observation, using the original IDs, selection-order
lists, structural missing ranks or graded category labels. It can be passed to
`FCData.from_frames`. Missing and unadministered observations are not generated
as effective events by this method.

Prior prediction does not fit the model or condition its parameter draws on
the observed response values. Inspect plausible response patterns and parameter
ranges before fitting. A fixed seed makes a run reproducible in a fixed software
environment; identical draws across dependency versions are not promised.

## Priors and identification

Family priors configure item roles. Defaults are `LogNormal(mu=0, sigma=2)` for
loading magnitudes, `ZeroSumNormal(sigma=1)` for ranking intercepts, and independent
`Normal(mu=0, sigma=1)` for graded intercepts. Global `trait_correlation` defaults
to `LKJCorr(eta=1)`. Trait means are zero and marginal variances are one, with a
centered multivariate Normal prior using the correlation Cholesky factor. Statement keying supplies the loading sign.

`prior_summary()` reports active priors, their labels, constraints and whether
they came from the user or defaults. Loading priors must have positive support;
ranking intercepts must retain their within-block zero-sum constraint. Unknown
distribution arguments and unsupported roles fail before sampling. Distribution
shape and model coordinates are managed by the model.

Block and statement overrides, parameter covariates and arbitrary user-defined
distribution factories remain planned extensions. `block_families` currently
accepts the same MGPPM configuration; distinct local prior configurations are
not implemented.

## Fitting, summaries and posterior prediction

```python
fit = model.fit(
    draws=1000,
    tune=1000,
    chains=4,
    cores=1,
    random_seed=42,
    target_accept=0.9,
)
print(fit.summary("item_parameters"))
print(fit.summary("traits"))
print(fit.diagnostics("item_parameters"))
replicas = fit.posterior_predictive(random_seed=43)
one_replica = fit.response_frame(replicas, draw=0)
```

These sampling lengths illustrate the interface; assess diagnostics and precision
for the actual analysis. The result exposes `fit.idata` as the native DataTree,
`fit.spec` as statistical metadata and `fit.sampling` as executed settings,
software versions and elapsed time. When no seed is supplied, the wrapper
generates and records a seed. Additional sampling settings such as `target_accept`
follow PyMC. `nuts_sampler="numpyro"` selects PyMC's optional NumPyro path after
installing `pyfcirt[numpyro]`.

`summary` and `diagnostics` delegate directly to ArviZ. Parameter roles are
`item_parameters`, `traits` and `trait_correlation`; `parameter_draws(role)`
returns labelled xarray draws. Constant constrained quantities, such as the unit
diagonal of a correlation matrix, have undefined sampling diagnostics; ArviZ may
report NaN or a warning for those entries. Interpret these separately from free
parameters. Tiny smoke-test chains are not sufficient for inferential summaries.

Posterior prediction freezes all free parameter draws and resamples the observed
response variables using PyMC's forward sampler. Derived quantities are recomputed
from those parameters rather than reused from stored deterministic values.
It returns a separate DataTree
and leaves the fitted posterior unchanged. Decoding preserves the same response
contract used for prior predictions. See the
[PyMC prediction API](https://www.pymc.io/projects/docs/en/latest/api/generated/pymc.sample_posterior_predictive.html)
and [ArviZ summaries](https://python.arviz.org/en/latest/api/generated/arviz.summary.html)
for the underlying interfaces.

## Native PyMC access

`built.model` is a native `pymc.Model`. Advanced users can use PyMC directly:

```python
import pymc as pm

with built.model:
    posterior = pm.sample(random_seed=42)
```

This is native PyMC sampling. The [model comparison](comparison.md) and
[Pearson/PPP](checks.md), [new-person scoring](scoring.md) and
[saving/rebuilding](io.md) interfaces are implemented. `built.spec.to_dict()`
provides JSON-compatible model metadata, including observation maps, priors,
variable roles, conditional person-block likelihood and a data fingerprint.

## Current model boundaries

MGPPM supports ranking, PICK and MOLE with block sizes 2, 3 and 4, including
different sizes in one model. Graded responses currently use two statements
and five declared categories. Graded and ranking fits are separate. Repeated
statements across blocks and blocks containing only one trait are rejected
until their identification constraints have been validated. Mathematical tests
and a small sampling smoke check do not establish parameter recovery or release
readiness. Controlled recovery is a separate, on-demand statistical validation
step when introducing a model or changing its statistical implementation.


## When chains do not mix

A completed sampler call and zero divergences do not establish convergence. Check
chain traces, R-hat, bulk/tail ESS and Monte Carlo error together. Inspect native
`sample_stats` for adaptation and tree-depth problems as well; the available
fields depend on the selected sampler.

If chains remain in different regions, preserve the run and investigate data
coding, identification, priors, initialization and adaptation before interpreting
summaries. Increasing the number of draws or using a GPU does not guarantee that
the chains will move between those regions. Record any revised sampling settings
or model assumptions explicitly and assess the new run separately.

Conditional new-person scoring can mix well even when calibration has not mixed.
Its diagnostics assess the conditional scoring calculations; they do not validate
the calibration posterior supplied to them.
