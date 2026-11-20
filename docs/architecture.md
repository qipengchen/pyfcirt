# Architecture

PyFCIRT prepares one runtime `PreparedInputs` and builds a reusable native PyMC model
from it. `Model.idata` is a separate `xarray.DataTree` of samples and results; graphs
and compiled functions are disposable runtime caches. See the [workflow](tutorials/tirt_workflow.ipynb) for analysis.

```text
Raw responses + design + model declarations
                   |
                   v
         data / PreparedInputs <--- formula coding
          frozen indices + constraints + observation origins
                   |
                   v
             native ModelGraph <--- scoring point or fitted PyMC priors
                 /           \
     fitted target            forward response graph
           |                        |
           v                        v
     native estimation       prediction + information
           |                        |
           +---------+--------------+
                     v
       Model.idata <--- validated xarray result groups
                     |
                     v
              summaries + scoring reuse
```

## Module handoffs

- `models/specification.py` owns static parameter roles, priors and model rules;
  distribution modules own their mathematical construction and response laws.
  Each measurement specification declares final parameter keys, logical axes,
  scope, regression eligibility and static construction flags. Option-dependent
  declarations resolve without response arrays or graph construction. Data preparation maps logical
  axes to actual coordinates and applies design-dependent identification once.
  Loading identification, residual reference values and format-dependent prior
  roles belong to those declarations; data owns component selection and records
  fixed provenance as each declared or design-dependent constraint is applied.
  These declarations are construction input, not a second saved layout schema.
  The specification associates each model with its native distribution, channel,
  response capabilities and any required category count. Population declarations use the same path for parameter axes,
  reference constants and scoring roles; data resolves actual group positions.
  Final-value domains and parameter bounds travel with the resolved layouts.
  Prior support remains a separate concept: a user prior can relax an ordinary
  orientation without removing a response law's probability or scale constraints.
- `data/design.py` owns the input declarations. `data/prepare.py` composes
  preparation and exposes its runtime views through `PreparedInputs`.
  `data/constraints.py` resolves identification and `data/coordinates.py`
  publishes canonical entries and relationships. Preparation validates external
  parameter points against those resolved constraints.
  One prepared batch view provides observation indices and event support to
  graph construction, result publication, response tables and simulation.
  Comparison batches use the prepared measurement pair catalogue for cutpoints.
  Data preparation supplies canonical entry indices and free-entry prior
  groups. Graph construction writes each native draw directly to its
  final entries. Model summaries use the same prepared groups.
  Sharing and identification consume the same prepared block membership.
- `covariates/design.py` owns formula encoding. `data/regressions.py` aligns that
  design to target rows and coefficient pools and resolves their unit usage.
  A partial replacement remaps the surviving bindings. `data/coordinates.py`
  publishes their entries and uses from those relationships. `covariates/model.py`
  and `trait_summary()` consume the prepared coefficient indices.
- `models/graph.py` and `models/assembly.py` create and retain actual graph
  bindings. Prepared batches supply likelihood kinds, names and observation
  slices. Checks can read saved probabilities without constructing the graph.
  The shared categorical binding applies the prepared permutation to both the
  likelihood and its derivative. Ranking evaluation obtains the marginal
  expression with `pm.logp` on the cloned native RV. Prediction and observed log
  likelihood reuse that expression through the same native forward evaluator.
  Assembly also resolves which final nodes to condition on for a requested
  output. Prediction, simulation and information reuse that selection. Final
  ability and population parameters share the graph's parameter lookup.
- `model.py` publishes complete configuration and result candidates. Prior and
  regression updates refresh parameter metadata and reuse the existing response
  metadata from preparation. Prediction reads the same frozen projection.
  `analysis/output.py`
  validates labelled arrays and pools likelihood factors by observation position.
  It assembles retained and replacement datasets in one native xarray DataTree
  before the model swaps the result reference. Ordinary fitting and `inference/gtum.py` share fitting
  eligibility, posterior publication and invalidation policy. Sampler-specific
  controls and transition mathematics remain in their respective estimators.
- `analysis/summary.py` selects and associates prepared parameter entries around
  native statistics. Prediction and information modules evaluate the existing
  graph. `scoring/` reuses preparation and construction with fixed final values
  or runtime priors estimated from selected PyMC variables.

## Data and construction

`FCInput` and `ItemInput` declare complete design and responses. Data preparation
normalizes formats, retains the original input-format label and one unit table
with complete scale/category metadata, resolves priors and freezes coordinates,
parameter slots, event supports and identification constraints once. `Model` can
then call `add_regressions()` with formulas and named covariate tables, or advanced
`Regression` mappings, to bind those frozen targets before inference. The covariate design
module owns Formulae coding; the data layer aligns targets and parameter pools.
Regression coefficients and conditional parameter nodes are created once. Each
response batch gathers final entries from its measurement slots and retains these
indices across prior and regression updates. Incremental prior and regression
updates publish affected parameter kinds and retain the measurement entries and
use rows that did not change. Scoring and simulation encode new batches after
aligning the new people. Models consume these prepared values directly.

Encoding produces aligned factor columns for each likelihood batch. The batch
uses these columns for native observed values, comparison positions and source
rows. The published observation identities follow the same likelihood order.
For augmented rankings, the ranking distribution defines the observed contrast
geometry. Encoding resolves distinct comparison patterns and gathers their geometry
in observation order.

Each measurement regression target retains its active flat indices and affected
units. Covariate construction gathers mean and scale at the active positions in
free-entry order, with one value per row for ordered cutpoints. Parameter
construction binds those vectors directly. The parameter-usage catalogue
publishes the coefficient-pool relationships to units and traits.
Population regressions keep their existing person/group design; they do not need
measurement memberships. Summaries own labels, scales and final table layout.

`ModelGraph.build()` owns assembly. `ParameterFactory` creates final free/fixed
parameter nodes, and `PopulationFactory` creates the population and abilities.
The nine measurement distribution classes own their response laws and generation
callbacks; parameter construction consumes the common prepared plan. Each public
`.dist()` returns an unnamed
native RV for use in a caller-owned graph. JAX kernels implement package numerical
formulas; native PyTensor/PyMC connects and registers the graph.

Prior construction resizes actual PyMC templates and binds transformed inputs
through their original operators. The constant description supports equality
and display. Construction and callback validation use the same native operator.
Initial and incremental configuration operations use the same short-lived prior
inspection context. It is discarded on success or failure; native parameter
binding and model lifetime do not acquire another persistent cache. PyMC
validates distribution arguments through its native graph. Prior and
regression updates use one selection of candidate parameter bindings to remove
superseded regressions and install their replacements.

## Reuse and independent scoring

`Model.build()` caches the base graph and returns its native `pymc.Model`.
Derived evaluation views reuse the cloned native observed responses. An
unregistered forward response is attached only when the fitting graph has
no corresponding observed response, as with augmented ranks and generation
schedules. The base graph retains the augmented fitting target and the
parameter expressions used for information and specialized estimation. Prediction,
likelihood and information read the same final parameter and response nodes. When
requested, fitting stores ordinary response log densities from those final
posterior nodes and leaves marginal ranking integration for later evaluation.
Information and information-based reliability share one response-parameter
program on the current graph. It retains one deterministic compilation and
uses the supplied parameter values at each call. Standalone prediction results also carry the prepared
unit lookup and event supports, which define comparable groups for PPC plots.

Scoring uses frozen unit and scale records while normalizing new responses.
Scoring and generation share the preparation step that replaces the person axis
and returns a complete parameter layout. Frozen calibration records are shared.
For a person-dependent parameter block, preparation copies the layout records
before binding the block to the new person axis. `score_model()` uses the same
graph assembly as calibration. PyMC replaces nodes with aligned final constants or runtime priors.
The retained output graph keeps the actual dependencies for scores, observations
and relevant output parameters. Both modes use `Model.fit()` and the standard
result publication. Approximation densities
and value transforms remain runtime objects. A reconstructed scoring round can
publish attached samples and derive the next approximation. Exposing or using
its native prior requires the original incoming approximation. MAP
points/moments remain explicit; only information and reliability have
non-MCMC analysis routes. MCMC prediction, PPP and comparison retain their own
sample requirements.

## Summaries and persistence

`Model.summary(details=False)` groups actual priors and constraints without a graph;
known population values are substituted directly. Prepared parameter blocks
provide their reporting component and final entry references; the summary groups
the existing entry memberships once. The detailed view expands IDs
and parameterization from the current runtime layouts. Specification text does not build
the per-parameter tables, MultiIndex relations or labels used by `coef()`.
`data_summary(by="model", details=False)` groups existing block/item counts by
model/response format and response law, deduplicating people. The prepared view
supplies declared answer counts separately from counts of encoded likelihood
categories. Block/item views and detailed
status/event counts are explicit options. `coef()` selects measurement or regression entries and
delegates posterior statistics to ArviZ with 95% HDI defaults. `trait_summary()`
summarizes per-draw marginal population moments under the current people's
covariate/group distribution using the shared population kernel, optionally by
group. These operations do not construct a graph or persist summary tables.
Fixed values have NA uncertainty and diagnostics.
Sample selectors use the original posterior chain/draw labels before moment
transformation, preserving label order and avoiding unselected draw calculations.
Native `sel` selects the labels, then `expand_dims` promotes any scalar sample
coordinates back to length-one axes without selector-type dispatch.
All-fixed summaries validate sample labels when supplied, then compute fixed
values once without creating sample diagnostics. Coefficient statistics use
ArviZ's xarray output on selected canonical entries. Display labels and formatting
are applied after those statistics are linked to the prepared membership relation.
Trait summaries apply native variable filters and coordinate selections to integer
output positions before allocating transformed posterior arrays. The existing
population kernels still define each draw's moments; only selected results cross
the device boundary and remain in host sample buffers. The bounded input chunks
retain the complete population parameter nodes needed for those moments.

Simulation draws from the graph's native forward view. Data preparation
decodes those draws directly to normalized rows and reuses the prepared
administration. The same encoder binds the new observed batches. It builds
observation origins from the local batches and publishes them with the batch
arrays in one merge of the frozen design. The recorded observation variables
let scoring and simulation remove the previous batch arrays directly. Public
generation schedules detach borrowed coordinates, population arrays and
the small population recipe at publication. Newly allocated event arrays are not
copied again. Schedule rows go from index validation to the shared encoder.
The encoder reads statement membership from the frozen design arrays. External theta points use one labelled person/trait validator in
data preparation; simulation and information retain their own point-source rules.

Summary entry points resolve native options and the output-column schema once,
then pass both to their table/array assembly. A small schema adapter handles empty
and all-fixed selections: native empty-input summaries do not support every median
mode. It provides names, never fabricated estimates or diagnostics. Posterior
flattening keeps integer entry positions without creating a full pandas MultiIndex;
display labels and the final presentation index remain separate from sample buffers.
When a prepared relationship only needs integer or numeric values, consumers read
native xarray variables directly. They do not construct a labelled DataArray and
scan every coordinate for each array access; labelled selection remains at the
sample-selection and presentation boundaries.

Save `model.idata` with native xarray, keeping construction code and data separately.
Rebuild the same model, repeat prior/regression declarations and call
`attach_samples(idata)` to reuse samples. Attachment checks final parameter dimensions
and coordinates; it does not recover declarations or infer likelihoods from a file.
Prepared layouts and Formulae encoders remain runtime objects, outside saved results.

Supplied samples must respect prepared fixed entries and shared loading identity,
as well as any complete fixed calibration point. Attachment checks these before publishing
samples. Evaluation accepts sampled final roots only when they retain stochastic
ancestors; graph constants remain constants. Explicit evaluation and simulation
points cannot change an already-fixed analysis's calibration conditions.

Item graph construction omits structurally inactive Q contributions using the
prepared active slots. The canonical loading vector and coefficient tables contain
active entries; the design retains the full item-by-trait Q relation. The graph
binds reduced trait and parameter inputs to the unchanged numerical kernels;
information projects their derivatives back to all traits.
