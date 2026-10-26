# PyFCIRT
Bayesian forced-choice IRT modeling in Python, built on PyMC and ArviZ.

## Development

Packaging, FCDesign, FCData, MGPPM likelihoods, simulation, Model/Prior and prior
prediction are implemented. Unified fitting, labelled summaries, ArviZ diagnostics
and posterior prediction are available. Both PyMC CPU and PyMC-dispatched NumPyro
CPU have passed small ranking and graded workflow checks. Conditional response
likelihood, PSIS-LOO, DIC, WAIC and Pearson/PPP checks are implemented; see the
[comparison guide](docs/comparison.md) and [predictive-check guide](docs/checks.md).
New-person scoring uses fixed calibration and a single trait fit;
see the [scoring guide](docs/scoring.md). [Saving and rebuilding](docs/io.md) is
implemented. MGPPM and new-person scoring now use centered trait parameters.
Earlier three-seed recovery results belong to the previous noncentered graph;
full empirical analysis and recovery are optional manual validation.
The [complete workflow source](docs/tutorials/mgppm_workflow.ipynb) and
[quickstart](docs/quickstart.md) are available as optional manual analyses.
The [recovery notes](docs/recovery.md) distinguish historical evidence from optional
future studies. Posterior MCMC is never run by formal pytest or CI.

## Implemented scope

- MGPPM ranking, PICK and MOLE support block sizes 2, 3 and 4. Mixed sizes
  require `FCDesign(..., allow_variable_block_size=True)`.
- MGPPM graded responses support two-statement blocks with an explicit
  five-category scale and direction. Ranking and graded models are fitted separately.
- Sampling uses native PyMC or optional PyMC-dispatched NumPyro. Centered
  ranking and graded GPU smoke checks passed. Full empirical analyses are manual
  tasks; GPU performance has not been benchmarked.
- New-person scores condition on fixed item parameters and trait correlation,
  retaining person/trait labels. Calibration uncertainty is not propagated.
- Global trait-correlation and family item priors are configurable within their
  identification constraints. Native PyMC graphs and ArviZ-compatible arrays are exposed.
- Saving/loading preserves data, draws and declarative specifications, with an
  explicit rebuild step. Custom native graph edits require separately maintained code.

Repeated statements across blocks, single-trait blocks, mixed ranking/graded fits,
local block/statement priors, item covariates and other model families are not
implemented. These boundaries describe tested API combinations. Complete tutorial execution
and population-recovery studies are optional manual checks, outside formal tests.
See the [data guide](docs/data.md), [model guide](docs/model.md) and
[simulation guide](docs/simulation.md) for working examples.
The [development roadmap](docs/roadmap.md) defines staged delivery and acceptance criteria.
The [requirements summary](docs/requirements.md) records the agreed capabilities,
and the [tutorial migration inventory](docs/tutorial_migration.md) distinguishes
existing local FC Tutorial functionality from new package requirements.

See the [architecture document](docs/architecture.md) for the planned package structure,
the boundary between FC-specific features and the PyMC/ArviZ ecosystem, and the model extension workflow.

The [input data design](docs/data_format.md) describes the proposed response and test-design tables.
Ranking input defaults to preference-order lists such as `[2, 3, 1]`, with an explicit
statement-rank long-table alternative. The [graded-response design](docs/graded_response.md)
defines category responses for two-statement blocks.

The [block composition design](docs/block_models.md) describes joint modeling with
different block sizes and measurement families, including full-ranking Thurstonian likelihoods.

The [core API design](docs/core_api.md) defines object responsibilities, block-size validation, and custom prior configuration.

The [parameter and extension design](docs/parameter_models.md) covers block/statement priors, item-feature regression, and reusable custom components.

The [basic scoring design](docs/scoring.md) defines new-person scoring with calibrated items.
The first version builds modules incrementally and completes a tutorial-style analysis
with MGPPM ranking and two-statement graded responses. Later families reuse that workflow
before adding more advanced features. Basic input includes full
rankings, PICK, and MOLE, with explicit handling of their structural missing ranks.
Graded responses belong to the basic input scope for the families that define them,
restricted to two-statement blocks with an explicit response scale.

The [model-comparison specification](docs/model_comparison.md) adds DIC and WAIC
to the first version. It requires fixed-draw numerical agreement with R2jags' default
chainwise DIC and legacy ArviZ 0.22.0 WAIC, alongside current ArviZ PSIS-LOO.

The [MGPPM test roadmap](docs/testing.md) maps implementation steps to concrete
test files. Prior predictive checks are a separate step after model construction,
before posterior MCMC, with dedicated tests for ranking and graded responses.
The plan follows selected Bambi testing practices, keeps pytest as the only new
test-tool requirement, and uses isolated R2jags / legacy ArviZ environments for
direct numerical comparisons on identical inputs.
