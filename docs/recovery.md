# Optional recovery investigations

Population parameter recovery is separate statistical validation that must be
requested explicitly. It does not run in ordinary pytest or CI. The default suite
retains fixed-parameter probability/gradient checks, observable-design checks and
labelled summary checks without
fitting a population model. Enable
long tests explicitly using `--run-recovery -m recovery`; see the
[testing guide](testing.md) for their exact scope and limits.

Defined or collected conditions are not completed runs. Check the saved sampling,
diagnostic and recovery reports before claiming a condition passed. Run each
condition through a bounded worker and preserve failures alongside any separately
planned extension. Ordinary pytest and code-coverage runs exclude actual MCMC.

## Interpreting recovery results

A passing check means that its predeclared numerical and convergence criteria
were met on that dataset. It does not establish nominal interval coverage for
every parameter, accurate tail estimates, or performance on arbitrary designs.
Inspect bias, RMSE, interval width and coverage for each parameter role; report
low ESS even when R-hat passes. Coverage calculated from a small set of correlated item
parameters is not an independent-binomial experiment.

Known ideal-point signs can inflate the correlation between signed locations
and their estimates. Report correlations of absolute locations and within each sign group too,
and interpret these with absolute errors and intervals. A narrow generating
range can make correlations unstable even when absolute errors are modest.
Fixed anchors and constant components must not pad recovery summaries.

A fixed zero population trait mean does not make a finite simulated sample's
empirical mean exactly zero. Shared trait-location error can affect several
intercepts or cutpoints together. Record empirical means when diagnosing that
pattern, but retain the original truth and recovery statistics. Do not recenter
truth or draws after seeing results to obtain a pass.

Original TIRT and GTUM decompose ranking inputs into known comparisons. Their
comparison marginals need not remain correct after conditioning on PICK/MOLE
selection, and products of pair probabilities are not normalized ranking laws.
Even a numerically successful composite fit can have biased item estimates or
intervals with low coverage. TIRTRank provides a joint Gaussian ranking target
for its supported homogeneous designs. Choose an appropriate target rather than
normalizing a pairwise product after fitting. A shared person-block observation
unit alone does not make information criteria from different likelihood targets
comparable. Inspect the PPP count unit as well: comparison checks count actual
comparisons, whereas categorical ranking checks count block responses.

There is a useful exception for GTUM data generated from independent unit-scale
Gumbel statement errors. At fixed person utilities, let `w_i = exp(utility_i)`.
For PICK, the pair `(i, j)` is observed precisely when either statement wins the
block. Conditional on that event, the probability that `i` wins is
`w_i / (w_i + w_j)`, matching the original logistic comparison probability.
Thus selection alone does not invalidate these PICK pair odds. This identity
does not establish identification, successful latent-parameter recovery, good
mixing, or calibrated posterior intervals for the product of comparisons.

MOLE also retains comparisons involving the least-preferred statement. Its
selection rule generally changes the conditional pair odds, even under the same
Gumbel utility law. Enumerating the complete ranking patterns and summing only
the genuinely observed pairs can distinguish this mismatch from an encoding
error. Keep such a failed recovery condition as evidence of the target's limits;
longer chains cannot by themselves repair a mismatch in the observation law.

## Planning an investigation

Before running a recovery investigation, specify the population, priors,
identification, seeds and error criteria beforehand. Resolve sign, scale, location
and label ambiguity before comparing estimates. Review uncertainty and sampling
diagnostics as well as point estimates. Do not require a broad multi-seed study
for an ordinary package change or silently rerun seeds until a criterion passes.

For the manual execution entry point and required deterministic checks, see the
[testing guide](testing.md). The [complete TIRT workflow](tutorials/tirt_workflow.ipynb)
is an optional substantive analysis, with its own diagnostic checks.
