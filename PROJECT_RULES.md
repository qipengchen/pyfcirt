# PyFCIRT project rules

Read together with the common engineering instructions in `AGENTS.md` or
`AGENTS.override.md`. These requirements specialize the package project.

## Scope and contracts

- Apply these rules to package code, tests and development work in this repository.
- Update both full and concise API contracts before changing public behavior.
  Keep engineering policy in the instruction files and statistical/data details
  in their authoritative contracts; do not duplicate their complete definitions.
- The parameter/coordinate contract is `docs/reference/parameter_contract.md`.
  The API index is `docs/api.md`; follow its links to the owning interface pages.

## Current module map and ownership

Paths below are relative to the repository root; implementation modules live in
`pyfcirt/`. This is the current source map and its responsibility contract, not a
claim that every legacy overlap has been eliminated. Check actual callers before
changing a boundary. Update this map in the same change when responsibilities move.

### Public interface and analysis lifetime

- `pyfcirt/__init__.py` exposes the public interface; it is not a second execution
  path. Subpackage initialization files provide package documentation/exports.
- `pyfcirt/model.py` owns Model configuration, lazy graph lifetime, sample
  attachment, verified recursive evidence, native fit orchestration and atomic
  configuration/result replacement.
- `pyfcirt/config.py` owns process device/precision configuration through native APIs.
- `pyfcirt/simulation.py` generates responses and records generation truth using
  the existing measurement distributions and prepared schedules.
- `pyfcirt/plotting.py` displays existing responses/checks/posteriors, reusing
  native plotting facilities; it does not fit models or invent probability laws.

### Data preparation

- `pyfcirt/data/prepare.py` owns `PreparedInputs`, its runtime views and preparation
  of initial, scoring and generation data, including final-parameter validation
  and the resolved runtime values for fixed population covariance.
  It also creates automatic simulation designs and labels fixed simulation arrays.
- `pyfcirt/data/design.py` owns input declarations and parses design notation.
- `pyfcirt/data/encoding.py` normalizes storage formats and encodes/decodes actual response
  events, category supports, batches and observation provenance. Prepared batches
  define likelihood kinds and names shared by construction, prediction and checks.
- `pyfcirt/data/constraints.py` applies model-owned identification/domain/prior declarations
  to the concrete design; resolves anchors, connected components and local blocks.
- `pyfcirt/data/coordinates.py` owns layout records and the production of canonical
  coordinates, parameter entries and gather/usage relationships.
- `pyfcirt/data/regressions.py` resolves ordinary-prior or regression ownership for each
  target, binds design rows/pools to entries, and prepares atomic replacements.
- `pyfcirt/data/grouping.py` aligns people to frozen population groups and reference labels.
- `pyfcirt/data/arrays.py` collects aligned arrays and explicit coordinate declarations
  before one Dataset construction. The temporary builder stays inside preparation.
- `pyfcirt/data/examples.py` loads bundled declarations/data and their study provenance.

Preparation publishes complete inputs. Construction and analysis consume their
indices and constraints instead of reprocessing raw tables or reselecting anchors.

### Model declarations, native construction and response laws

- `pyfcirt/models/specification.py` owns model/population declarations, parameter meaning,
  defaults, admissible roles, statistical constraints and reporting policy.
- `pyfcirt/models/priors.py` owns native template validation, subsetting and direct
  PyMC binding. Prior descriptions support equality and display. Data code selects
  concrete targets and validates external final values from the prepared entries.
- `pyfcirt/models/graph.py` owns canonical final parameter construction, likelihood
  gathers and native observation registration. Its categorical and Beta response
  builder binds the same model kernel arguments for inference and information.
- `pyfcirt/models/assembly.py` composes one native ModelGraph from prepared bindings.
  It owns native calibration interventions, retained graph outputs, the evaluation
  clone, response-expression rebinding, bounded deterministic point compilation
  and native-coordinate identity.
  It consumes prepared data.
- `pyfcirt/models/population.py` owns ability nodes and population mean/covariance laws.
- `pyfcirt/models/categorical.py` owns shared stable categorical log densities, response
  sizing and aggregation of complete patterns into declared events.
- `pyfcirt/models/dist.py` exports the public native measurement distributions.
  Each family owns its underlying distribution implementation.
- `pyfcirt/models/likert.py` owns DichotomousIRT, GPCM and GGUM response distributions.
- `pyfcirt/models/sequential.py` owns TwoPLRank/GGUMRank distributions and their shared
  successive-choice calculation; endorsement differences remain model-specific.
- `pyfcirt/models/mgppm.py` owns the MGPPM response distribution.
- `pyfcirt/models/thurstonian/tirt.py`, `pyfcirt/models/thurstonian/gtum.py` and `pyfcirt/models/thurstonian/rank.py` own TIRT, GTUM and TIRTRank
  response laws, respectively. The ranking law owns order-region geometry and
  separates augmented fitting from generation and lazy marginal event probabilities.
- `pyfcirt/models/thurstonian/normal.py` owns shared comparison response construction,
  interval probabilities and Gaussian region integration.
  Gaussian region integration uses the native JAX inverse log-CDF.

### Covariates

- `pyfcirt/covariates/specification.py` owns Regression declarations/options and formula
  syntax, independently of concrete measurement target indices.
- `pyfcirt/covariates/design.py` fits/applies Formulae encoders and training transforms.
- `pyfcirt/covariates/pairwise.py` owns the normalized pairwise group prior and its RNG.
- `pyfcirt/covariates/model.py` creates each prepared regression's mean/scale expressions
  once, then routes them to target nodes. It does not refit feature encoders.

### Results and analysis

- `pyfcirt/analysis/output.py` owns validated posterior access, final
  parameter-point extraction and atomic result publication. It attaches prepared
  identities and replaces posterior-dependent results.
- `pyfcirt/analysis/prediction.py` evaluates retained expressions for prediction
  and pooled marginal response log likelihood.
- `pyfcirt/analysis/summary.py` presents data/model/response summaries and coefficient
  statistics using existing entries, usage relationships and native ArviZ.
- `pyfcirt/analysis/dif.py` contrasts respondent profiles using the fitted regression encoders
  and posterior measurement parameters.
- `pyfcirt/analysis/traits.py` summarizes final population/trait moments with current-person
  and group context; it does not re-estimate model parameters.
- `pyfcirt/analysis/checks.py` computes Pearson PPP and residual discrepancies from the
  existing probabilities and replicas, respecting observation/group indices.
- `pyfcirt/analysis/comparison.py` provides WAIC and the declared DIC using existing
  response log likelihood; it does not reinterpret likelihood observation units.
- `pyfcirt/analysis/information.py` computes Fisher information from distribution kernels
  and aggregates using prepared observation relations.
- `pyfcirt/analysis/reliability.py` computes empirical and information-based reliability
  from existing ability uncertainty/information.

Native xarray handles persistence. Saved metadata is an identity/constraint
projection, not a serialized live graph, prior operator or Formulae encoder.

### Scoring, shared support and specialized inference

- `pyfcirt/scoring/scoring.py` coordinates runtime posterior approximation and
  independent point or posterior scoring models. It supplies remaining native
  RV means for the fixed complement and passes Model-verified respondent exclusions
  to data preparation. It reuses prior binding and ordinary Model.fit.
- `pyfcirt/common/validation.py` owns small identifier/frame primitives shared by input
  preparation and feature encoding; it is not a second model validator.
- `pyfcirt/scoring/approximation.py` fits per-kind densities in PyMC's
  unconstrained coordinates and applies them to native graphs at runtime.
- `pyfcirt/inference/conditional_mode.py` owns regional proposals and native independence-MH.
- `pyfcirt/inference/task.py` owns continuous native BlackJAX task state, warmup and chains.
- `pyfcirt/inference/runner.py` runs those tasks and converts native outputs; publication
  policy remains with the common result owner.
- `pyfcirt/inference/gtum.py` supplies the specialized experimental GTUM applicability and
  proposal geometry. Ordinary model estimation remains in Model.fit/native PyMC.

### Supporting project areas

- `pyfcirt/_example_data/` contains bundled data/provenance resources, not an
  alternative preparation or statistical implementation.
- `tests/` owns deterministic behavioral/numerical regression checks; ordinary
  tests do not run calibration/scoring MCMC.
- `docs/` owns public API/data contracts, mathematical explanations and executed
  user examples. `scripts/` owns explicit developer/validation workflows.
- /pyfcirt/ owns the R interface and conversion to R analysis objects. Python
  remains responsible for preparation, model semantics, inference and persistence.
- .github/workflows/ owns CI automation; it does not define model semantics.

Do not add a generic utility, registry or parallel schema when an owner above
already provides the operation. Different mathematical tasks may legitimately
share arrays while retaining separate numerical implementations.

## Statistical and implementation requirements

- Use PyMC for distributions, priors, transforms and estimation; ArviZ for
  established posterior statistics; xarray for labelled selection/alignment and
  storage. Do not reimplement these in package-specific dispatch layers.
- Use JAX for statistical numerical kernels and derivatives. Host-side NumPy,
  pandas and xarray are appropriate for input/index preparation, I/O and display.
  Avoid unnecessary device transfers and parallel NumPy/JAX versions of a law.
- Kernels receive arrays, explicit configuration and any random key, and return
  values/state/diagnostics. No file access, logging, hidden random state or
  mutation of a captured model inside a numerical kernel. Bind static choices
  once at construction rather than reinterpreting them per evaluation.
- Vectorize independent observations with array operations or `vmap`. Use JAX
  control flow for recurrences and data-dependent branches. Static short loops
  are acceptable when clearer; do not allocate a huge broadcast just to remove
  a loop. Split independent random operations with explicit keys.
- Keep canonical parameter kinds and prepared integer references throughout
  construction and output. Use auxiliary coordinates and usage relations for
  identity. Do not parse namespace strings, joined labels or printed tables.
- Select requested entries/draws before materializing large posterior arrays.
  Compute shared-entry statistics once, then associate them with units. Attach
  output identities from the prepared plan, not a second map rebuilt by analysis.
- Validate external boundaries; trust validated internal handoffs. Fail clearly
  for unsupported contracts. Do not guess IDs, reorder external samples, repair
  missing metadata, silently clip probabilities or add speculative fallbacks.
- Preserve normalization, sign/scale conventions, support and observation units.
  Augmented fitting densities and marginal response probabilities are different
  targets. Keep expensive ranking-region integration outside calibration NUTS.
- Use descriptive names, modern type annotations on public interfaces and
  concise English docstrings/comments. Explain non-obvious mathematics, axes,
  numerical constants and tradeoffs; avoid comments that restate an assignment.

## Validation and environment

- Extend existing focused tests where possible. Check densities and needed
  gradients, fixed/shared values, typed coordinates, category direction, state
  isolation and affected downstream workflows. Supplied arrays are not fits.
- Ordinary tests must not run calibration/scoring MCMC. A full suite, actual
  MCMC or recovery experiment requires an explicit user request.
- Separate construction, compilation/first-call and hot-call timing; compare
  matched inputs/configurations and check memory/storage when relevant.
- Never invent numerical outputs, convergence, equivalence or coverage. Keep raw
  records, fits and machine configuration outside version control.

Use Python 3.12+ and an existing suitable environment. Follow the repository's
Ruff settings and 100-column style. For example:

```sh
python -m pytest tests/data/test_encoding.py -m "not slow and not reference and not recovery"
python -m ruff check pyfcirt/data/encoding.py tests/data/test_encoding.py
python -m ruff format --check pyfcirt/data/encoding.py tests/data/test_encoding.py
```

## Pull requests and review

- Reuse the agreed branch/worktree. Never reset, stash or discard unrelated work
  to prepare delivery. Do not commit secrets, credentials, private data, generated
  fits, local overrides or execution logs.
- A side chat completes its scoped work, updates contracts, runs affected checks,
  commits the task's changes, pushes the branch and creates/updates a draft cloud
  PR. Then hand off the exact revision and stop editing until ownership returns.
- Write PR titles, descriptions and comments in English. Use `Purpose` for the
  section that explains why the change is needed. Describe what changed and useful
  before/after behavior using verified examples. Do not emphasize agent identity,
  narrate tools or add a separate limitations section. Report validation gaps
  honestly in the handoff and relevant review notes.
- Parameter-contract PRs must show an actual posterior `idata` tree, variable
  dimensions, representative auxiliary coordinates, fixed/shared metadata and
  pooled likelihood observation origins. State the input and actual run that
  produced it; do not present a planned structure as executed output.
- The main chat reviews the exact PR revision locally, verifies/fixes findings,
  runs relevant checks and repeats review. Cloud review is additional when
  requested. Describe the review mechanism/model only when actually known.
- Task-scoped commits, branch pushes, draft PR updates and verified review
  comments are authorized by this workflow. Never merge/push changes directly to
  `main` or enable auto-merge without explicit user approval of that revision.
- Before an authorized merge, squash the PR's commits into one coherent commit.
  Do not rewrite unrelated branches or discard uncommitted work. Committing or
  updating a PR is not authorization to merge it.

Review findings must identify a concrete trigger, consequence, severity and
file/line. Check numerical targets, data/graph/result/scoring boundaries and actual
deletion of duplicate rules. Leave mechanical formatting to Ruff.
