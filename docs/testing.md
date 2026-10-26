# Testing PyFCIRT

Formal pytest tests do not run posterior MCMC. Sampling checks are explicit,
manual developer actions outside `tests/`. This keeps package regression checks
separate from judging convergence in a substantive analysis.

## Required automated checks

- **Data:** design relationships, labels, ordering, variable block sizes, selection
  lists, explicit statement ranks, graded scales and structural PICK/MOLE NA.
- **Mathematics:** fixed-parameter likelihoods, event normalization, gradients,
  identification constraints and generation for ranking sizes 2/3/4 and graded pairs.
- **Construction and priors:** one shared trait structure, centered parameters,
  supported prior overrides and prior predictive support. Forward prior draws are
  inexpensive and do not involve a posterior sampler.
- **Inference wiring:** mock `pymc.sample` when checking options, return groups,
  coordinates, metadata and error handling. An autouse fixture rejects accidental
  real calls to `pymc.sample`; an individual test can supply its own mock.
- **Results:** use small fixed arrays or explicitly supplied saved traces for
  summaries, prediction, Pearson/PPP, conditional log likelihood and continuation.
- **Scoring:** compare conditional densities and gradients to independent formulas;
  use independent target densities and mocked draws to verify fixed calibration
  and labels. Unanswered-person scores use exact prior
  draws, not MCMC.
- **Comparison:** fixed-draw DIC and WAIC calculations agree numerically with pinned
  R2jags post-processing and isolated ArviZ 0.22.0. Neither reference fits a model.
- **Distribution and docs:** wheel/resources, outside-checkout import, persistence,
  API documentation and small fresh-kernel runner fixtures.

Keep these tests in their existing natural modules. Shared construction-only
builders live in `tests/_workflow_mgppm.py`; existing population-design checks in
`tests/recovery/` do not fit any models. Do not match another project's test count
or build a Cartesian product of samplers, seeds and every supported data shape.

## Commands

```shell
python -m pytest -m "not slow and not reference"
python -m pytest -m reference
python -m pytest -m slow
python -m pytest
```

`slow` now means subprocess notebook-runner checks or continuation using supplied
traces; it does not enable MCMC. Reference tests skip only when their separate
reference executables are not configured. Saved-trace tests skip when inputs are
not supplied. A skip is never numerical agreement or sampling evidence.

## Optional manual sampling

```shell
python scripts/check_mgppm_sampling.py --task rank --sampler numpyro --output /path/to/new-check
```

Use the storage launcher when configuring external data/results/caches. NumPyro
is optional; choose `--sampler pymc` for native PyMC. Configure a GPU through the
worker environment and record the actual device. The script uses 30 warmup and
20 retained draws on two chains; it checks finite outputs and prediction wiring,
not convergence, uncertainty accuracy or population recovery. It is never invoked
by pytest or CI. The centered implementation already has actual small sampling
records for both tasks and scoring modes; moving their checks out of pytest does
not require rerunning those fits.

The full tutorial remains a user-facing analysis example. Running it on empirical
inputs, reviewing its diagnostics, or conducting multi-seed recovery are optional
manual investigations, not package-test or v0.1 acceptance gates. Preserve failed
and cancelled runs without describing them as successful analyses.

## Design reference

Borrow Bambi's use of construction tests, sampler mocks and reusable fitted
results, rather than reproducing its entire suite. Bambi also has actual sampling
fixtures; the current PyFCIRT policy deliberately keeps posterior sampling outside
formal tests. See [Bambi tests](https://github.com/bambinos/bambi/tree/main/tests).
