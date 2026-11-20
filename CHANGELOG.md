# Changelog

## 0.1.0b1

- Prepare the first public beta, including CPU distribution and installation checks.

## Input names and response formats

- Create `Model()`, add response tables with `add_fc()` and/or `add_items()`,
  then call `prepare()` once. Fit and analyze on that object.
- Use `model`, `response_format` and `scale` for forced-choice inputs and
  `model` and `categories` for individual items.
- Use `BlockSpec` and `ItemOverride` to set options for individual blocks and
  items. `response_format` describes the questionnaire task, and
  `ResponseTable.format` describes table storage.
- Prepared event metadata and response support records use `response_format`.
  Data/coefficient summaries use `model_response_format` instead of `model_task`.
- Historical dataset resources retain their original schema and checksums; the
  bundled reader maps their recorded field to current declarations.


## Prepared input and parameter naming

- `PreparedInputs.from_inputs(...)` replaces `prepare_data()` and `PreparedData`,
  accepting optional formula strings, feature tables and regression options before
  `Model` builds its graph. Later `Model.add_regressions()` uses the same binder.
- Use `cutpoints` for TIRT/GTUM prior overrides, matching `fc.cutpoints` selectors.
- Regression options are `standardize_predictors`, `deviation_noncentered` and
  `parameter_residual_sd`; population scaling is `theta_residual_scale`.
  Coefficient and ability parameterization options retain their distinct names.
- Defaults, native posterior variable names and latest-definition-wins behavior
  are unchanged. Saved draws attach after rebuilding the matching declarations.

## 0.1.0.dev0 — Unreleased

- Require JAX 0.12.1 and use its native inverse log-normal CDF for marginal ranking integration.

This is a development version. Implemented interfaces, targeted validation and
final distribution acceptance have separate scopes; see the
[development status](docs/roadmap.md).

- Implement six FC models: MGPPM, TwoPLRank, GGUMRank, TIRT, TIRTRank
  and GTUM. Keep categorical ranking laws distinct from products
  of comparison likelihoods. Add native single-item MIRT 1PL–4PL, MGPCM and
  distance MGGUM with their own supported designs and identification constraints.
- Provide explicit long-format FC and single-item data contracts, labelled
  designs, structural PICK/MOLE unknowns, administration masks and supported
  variable block sizes/scales. `FCData.from_wide()` accepts pair-wide input;
  `FCData.from_block_wide()` accepts preference-order lists for binary
  RANK/PICK/MOLE response formats. Rank projection rejects cycles rather than inventing
  a complete order; comparison models can retain cyclic responses.
- Separate `response_response_format` and `block_type` on the design from `response_format`
  and `layout` in `PreparedInputs.from_inputs()`. Response formats and models accept a global value or
  an ordered list, with partial block overrides. Support multidimensional 1PL
  designs with fixed active loadings and estimated trait correlations.
- Align active FC prior defaults with the declared PyMC specifications. In
  GTUM/GGUMRank, location sides are explicit; nonzero sides use signed
  TruncatedNormal(2, 1, lower=0) magnitudes without a finite upper bound.
  TIRT selects residual reference constraints from the complete block design,
  including supported shared statements. An optional noncentered statement-effect
  parameterization preserves the physical model and centered abilities.
- Separate native likelihood components from parameter ownership. Caller-owned
  PyMC graphs can provide priors, parameter covariates and shared abilities for
  FC/single-item composition. Supported mixed FC block assignments are
  experimental and require `experimental=True`.
- Provide native PyMC and BlackJAX inference, an optional NumPyro route, prior and
  posterior prediction, labelled ArviZ summaries/diagnostics, conditional
  likelihood, compatible LOO/DIC/WAIC, categorical PPC and versioned persistence.
- Retain fixed-calibration new-person MCMC scoring and add supported MAP/EAP
  methods with explicit concavity, integration and dimension limits. EAP uses
  QMC by default, with explicit Gauss-Hermite integration or a Laplace QMC
  proposal where supported. These methods do not propagate calibration
  uncertainty.
- Add unified `simulate()` generation from priors, fixed parameters or saved
  posterior draws, conditional information, known-parameter joint TIRT ranking
  information, defined reliability coefficients and Pearson posterior predictive
  checks. Posterior predictions and checks retain labelled observations and can
  be saved as plain ArviZ/xarray data.
- Remove the experimental `m2()`, `moment_discrepancy()` and `ld()` interfaces,
  including their RMSEA, CFI and TLI outputs.
- Add separate experimental GTUM conditional-mode IMH and TIRTBeta/GTUMBeta
  continuous comparisons. Their compatibility restrictions and validation limits
  remain explicit. TIRTRank provides a separate Gaussian ranking implementation;
  GTUMRank/GTUMSelection and the reserved TIRT extensions remain paused.
- Default to float64 and native PyMC on CPU. Add process-wide `set_backend()`
  for explicit BlackJAX/device selection and `set_float_precision()` for new
  PyTensor/JAX floating defaults; existing arrays/graphs and integration
  tolerances remain separate.
- Bound default prior and result displays while retaining complete tables,
  diagnostic extremes and explicit detailed views. Save and load fixed MAP/EAP
  estimate datasets through the common persistence interface.
- Provide independent numerical reference fixtures without an R runtime
  dependency. Ordinary tests prohibit posterior MCMC; bounded manual sampling
  and explicitly enabled recovery remain separate. Add an isolated-environment
  installer, API documentation and executed source examples; final installed
  examples, statistical acceptance and whole-suite coverage are distinct gates.
- Add `--no-download-cache` to the environment installer for constrained
  temporary disks, retaining the default shared-download cache when omitted.

See the family guides and [persistence](docs/io.md) for saved-version behavior.
This record describes the current unreleased interfaces, not a completed release,
validation of every model/format/backend combination or a publication claim.
