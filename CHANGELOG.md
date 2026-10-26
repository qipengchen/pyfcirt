# Changelog

## 0.1.0.dev0 — Unreleased

- Add `FCDesign` and `FCData` with explicit statement/block/trait relationships,
  preference-order lists, structural PICK/MOLE missingness, administration states,
  labelled indexing and variable block sizes.
- Implement MGPPM ranking, PICK and MOLE for sizes 2, 3 and 4, and a separate
  five-category graded-pair model.
- Add constrained prior configuration, native PyMC models, prior and posterior
  predictive sampling, and native PyMC or PyMC-dispatched NumPyro CPU inference.
- Add labelled summaries, ArviZ diagnostics, categorical Pearson/PPP checks,
  conditional response log likelihood, PSIS-LOO, R2jags-style pV DIC and WAIC
  numerically checked against ArviZ 0.22.0.
- Add fixed-calibration new-person scoring, including
  exact prior prediction for people without answers.
- Use centered multivariate-Normal traits in calibration and new-person scoring.
  Saved model version 2 records this graph; version 1 draws remain readable,
  while rebuilding that earlier graph is explicitly rejected.
- Remove posterior-mixture scoring and repeated conditional fits. Score results
  now retain one native posterior and use score bundle version 2; older score
  bundles are explicitly rejected. Calibration-fit bundles remain unchanged.
- Add checksummed JSON/NetCDF persistence and explicit model reconstruction.
- Add mathematical, encoding, saved-trace integration and external-reference checks.
  Formal tests prohibit posterior MCMC; provide a separate manual sampling script.
  Historical recovery evidence belongs to the noncentered graph. Posterior MCMC
  and population recovery are now optional manual validation outside pytest.
- Add generated API documentation and a complete workflow notebook source.
  Full empirical notebook execution is optional manual analysis; this is not a
  released version.

Other FC families, local priors, item covariates, mixed-family fitting and GPU
performance validation are outside this initial implementation.