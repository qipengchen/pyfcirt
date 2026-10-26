# PyFCIRT development roadmap

PyFCIRT is under active development. The implementation status below separates
implemented modules from their evidence and optional research validation. The first release targets one complete, usable model rather than
partial implementations of several families.

Implemented so far: package setup, statistical specification, FCDesign/FCData,
MGPPM event likelihoods and simulation, Model/Prior, native PyMC construction and
prior prediction, unified fitting, labelled parameter summaries, ArviZ diagnostics
and posterior prediction. PyMC CPU and PyMC-dispatched NumPyro CPU passed small
ranking/graded workflow checks. See the [data guide](data.md), [model guide](model.md)
and [simulation guide](simulation.md). Conditional likelihood, PSIS-LOO, DIC, WAIC
and Pearson/PPP are implemented; see [comparison](comparison.md) and [checks](checks.md).
New-person scoring with fixed calibration and save/load/rebuild
are implemented; see [scoring](scoring.md) and [persistence](io.md). The shared model and scoring graphs now use centered trait parameters.
Earlier three-seed recovery results were generated with the previous noncentered
graph. Centered recovery and complete empirical tutorial runs are optional manual
investigations, outside formal tests and v0.1 acceptance.
Centered ranking and graded NumPyro GPU smoke checks have passed.

## First release: MGPPM

The implementation sequence has these module milestones:

1. **V1-01**: installable package, development environment and distribution checks.
2. **V1-02**: statistical specification, identification and data provenance.
3. **V1-03**: FCDesign and statement/block/trait relationships.
4. **V1-04**: FCData, preference-order lists, PICK/MOLE missing ranks and graded categories.
5. **V1-05**: labelled indexing, observation encoding and block-size groups.
6. **V1-06**: MGPPM likelihood, gradients and response generation.
7. **V1-07**: Model, Prior and the native PyMC graph.
8. **V1-07a**: prior predictive generation before posterior fitting.
9. **V1-08**: fitting, labelled results and ArviZ diagnostics.
10. **V1-09**: item/person parameter summaries.
11. **V1-10**: posterior predictive responses.
12. **V1-11**: response log-likelihood, PSIS-LOO, DIC and WAIC.
13. **V1-12**: FC-specific posterior predictive checks and PPP.
14. **V1-13**: new-person scoring with fixed calibration.
15. **V1-14**: saving, loading and explicit reconstruction for continued analysis.
16. **V1-15**: MGPPM tutorial source, generated API docs and lightweight runner checks.
17. **V1-16**: clean-install, deterministic workflow and reference-numerics acceptance.

MGPPM ranking and two-statement graded responses each expose the end-to-end analysis workflow. Formal tests use mocks or saved
traces; actual sampling is an optional manual check.
Ranking includes PICK/MOLE through marginal event probabilities and supports
validated variable block sizes. Graded responses have a separate category-score
definition and identification constraints. A data container accepting a task does
not by itself establish that a model supports that task.

See the [requirements](requirements.md), [testing specification](testing.md),
[scoring targets](scoring.md) and [model-comparison definitions](model_comparison.md)
for the acceptance criteria. Planned capabilities are not evidence of support.

## Subsequent development

Add 2PLM-RANK, GGUM-RANK, TIRT and GTUM one at a time, each completing the common
workflow and its own mathematical checks. Family-specific graded support must
have a defined likelihood and evidence.

After the core families, consider local priors, mixed-family models, parameter
covariates and reusable custom components. Keep the implementation small and reuse
PyMC and ArviZ functionality. Large simulation infrastructure, a Stan runtime and
publication preparation are outside the first release.
