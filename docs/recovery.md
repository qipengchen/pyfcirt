# Optional recovery investigations

Population parameter recovery is a research or manual validation activity. It is
not part of formal pytest, CI, or the v0.1 acceptance gate. The formal suite retains
fixed-parameter probability/gradient checks, observable-design checks and labelled
summary checks without fitting a population model.

## Historical development evidence

Earlier noncentered development runs used three fixed seeds (11, 29, 47),
180 calibration people, 30 new people and 18 blocks with three traits. Ranking
included sizes 2/3/4 and rank/PICK/MOLE events; graded used five-category pairs.
Their checks covered calibration and fixed/posterior-mixture new-person scoring.
Those results belong to that earlier graph and do not establish recovery of the
current centered graph. The original scripts and run metadata are retained in
external development evidence, outside the formal test suite.

The centered graph has actual small native PyMC and NumPyro GPU execution checks
for both branches, with separate scoring and persistence checks. Short chains
show execution, not convergence or recovery. The long centered empirical run was
cancelled at the owner's request when the testing policy was simplified; no
successful full empirical analysis is claimed.

If a future model investigation needs recovery, specify the population, priors,
identification, seeds and error criteria beforehand. Resolve sign, scale, location
and label ambiguity before comparing estimates. Review uncertainty and sampling
diagnostics as well as point estimates. Do not require a broad multi-seed study
for an ordinary package change or silently rerun seeds until a criterion passes.

For the manual execution entry point and required deterministic checks, see the
[testing guide](testing.md). The [complete tutorial](tutorials/mgppm_workflow.ipynb)
is an optional substantive analysis, with its own diagnostic checks.
