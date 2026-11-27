---
orphan: true
---

# Reference

See the [current page](api.md).

Create `model = Model()` and add tables with `model.add_fc()`,
`model.add_items()`, or both. Select the measurement law with `model=`. Set
`response_format` and `scale` for forced choice or `categories` for individual
items. Use `blocks` and `items` to override options locally. Call
`model.prepare(**options)` once to align the added tables. Alternatively,
pass existing input declarations or prepared data directly to `prepare()`. Use `model.add_regressions()` and
`model.set_priors()` for optional configuration before fitting. Call `model.fit()`,
`model.coef()` and `model.posterior_predictive()` on the same object. `fit()` returns the result
DataTree, also available as `model.idata`. Call `model.score()` to prepare and
fit a separate scoring model, or `model.score_model()` to prepare it for
estimation later. A new single-channel response DataFrame needs `format=...`.
For experimental recursive scoring, fit the first calibration with
`model.fit(retain_calibration=True)` and call `model.score(..., calibration="posterior")`.
The first round updates all supported free shared variables. To update a subset,
pass an explicit prior built with `posterior_approximation(var_names=...)`.
Score the next batch from the scored model to reuse its selection. Use `build()`
to access the native PyMC model.
For experimental DIF, call `model.add_dif_terms()` before fitting. Use
`model.dif_summary()` for intervals or pass one two-group TIRT `contrasts` mapping
with `bayes_factor=True` for per-parameter evidence computed from the model's DIF priors.
For continuous priors, the result includes ArviZ's BF values and uses its native warnings.
Use native PyMC compound sampling for experimental `prior="spike_and_slab"` terms.
Import distributions for custom PyMC models from `pyfcirt.models.dist`.
The preparation entry owns the layout and encoding order. It validates the
typed axes of wide inputs before expanding response rows. The measurement
catalogue supplies the frozen design identity, before publication adds people
and observations. Prepared inputs and borrowed result metadata are read-only;
use `.copy(deep=True)` for edits. `data_coverage(details=True)` shows declared
answer categories and likelihood-factor categories separately.

Fixed scoring conditions the shared native graph on the prepared final
parameter snapshot. The published `idata` group records that snapshot.
Posterior scoring replaces selected shared priors and fixes the other retained
shared variables in the same PyMC intervention. The native graph keeps the
dependencies of the observed factors, ability and reusable parameter outputs.
`Model` checks that the supplied approximation covers its committed person and
batch IDs. Preparation uses the verified IDs to reject repeated respondents
before aligning predictors or encoding observations. Failed checks leave the
existing result available.
Scoring and generation align each new person's group and theta covariates to the
same row, then reuse the frozen feature encoding and group order. Person-local
gathers are reused when the person axis retains its length. Resizing it
rebuilds the affected kinds and gathers.
`PopulationSpec` validates group options. Preparation resolves actual group labels.
Generation for new people reuses the validated calibration point. Preparation
checks that fixed covariance is compatible with the declared correlation axes
and supplies the derived population values to the graph.
Preparation retains measurement specifications in unit order and native batches
in observation order. Each batch exposes its shared observation indices to the
graph, result publication, response display and simulation. Measurement slots
retain the binding and gather axis. Direct comparisons use one local pair row
for cutpoints and the global pair index.
Posterior simulation selects and extracts a joint draw from the same validated
posterior dataset. Recursive calibration uses one validated posterior for its
prior approximation and fixed complement. Both use the prepared point variables.
Final-parameter validation aligns shared axes once per Dataset. Information
reuses the verified axes of an extracted parameter point and applies the same
value constraints as an explicit point. The runtime graph reuses deterministic
response-parameter compilation across points with matching shapes and settings.
A supplied ability array is aligned independently and can replace theta
when the posterior did not retain it. Dependent bounds also use retained fixed
calibration values. Simulation checks generated truth through the same validator.
Identical defaults share template preparation. Native parameter bindings own independent draws.
Slots refer to parameter owners through their binding IDs. Model labels come
from the prepared unit axis. Prior and regression updates retain existing measurement gathers and entry
arrays for kinds whose layout and constraints are unchanged. When regression
membership changes, preparation updates its rows in the parameter-use catalogue
and retains the measurement rows. Coefficient and residual-scale priors keep
their own bindings even when they share a PyMC template. Measurement regressions
hand off aligned free positions, feature rows and coefficient pools.
The pool axis carries the target kind, trait association and scale.
Preparation reads static PyTensor shape and parameter information directly.
PyMC folds symbolic constant hyperparameters and checks native priors on the actual graph. Preparation checks the axes and declared domains of its prior
groups. Prior selection, construction and summaries reuse those groups. A
measurement slot with a full local range uses its existing integer axis. Coefficient centering
is resolved by term before graph construction.
Native 4PL truncation uses the canonical lower-entry indices and the base
prior's native transform endpoints. Ordered and affine bindings read the
base prior endpoints once and retain live affine scale and offset inputs.
Local fixed-value views are gathered on demand.
Users select the support of custom priors for scales. Noncentered affine
expressions feed their final nodes directly.
Recursive calibration uses one native dependency relation to select eligible
RVs and determine the fixed complement. Scoring collects native RV means while
the common point reader supplies final parameters. The scoring model reuses its native
source graph with the existing fixed values applied. It reads already untransformed
coordinates directly and uses PyMC value transforms for the others. Native RV
dimensions order the labelled draws. Fitting and prior replacement use the same
resolved transforms, with optional user overrides by RV name. A stochastic bound supplies paired draws to the
bounded variable's transform without requiring a density fit for the bound.
The default is one joint approximation across automatically selected
shared parameters. `method="auto"` selects the density family. Ordinary
idata retains samples, fixed calibration values and evidence. The approximation lives in memory.
Scoring validates calibration values against the frozen reference catalogue.
Reuse requires matching prepared shared entries, structural fixed values
and frozen measurement predictors. Batch verification also includes the fixed
calibration snapshot. Retain `calibration_evidence` to restore recursive results
and fit the next approximation. If the group is missing, attachment requires the
original runtime prior and explicit `assume_same_batch=True` confirmation.
Recorded incoming fit identifiers are checked against the active prior.
Native calibration draws also carry a coordinate identity. Reusing them requires
matching the shared parameterization and native bounds. For untagged draws,
`assume_same_batch=True` also affirms this parameterization.
Building, fitting or simulating from this round's prior requires its original
incoming approximation. Graph construction and
reporting use the same prepared coefficient indices and training feature encoding.
`PreparedInputs.parameter_layouts` provides read-only lookup by parameter kind.
Ordered groups are part of the entry catalogue. Dependent bounds point to the
completed lower-entry indices.
Native conditionals receive the means and scales of their active entries or
ordered rows. Partial replacement compacts the affected regression axes together before
publishing. Each scope uses one pool remap for target and unit references,
local axes and prior batches. Surviving entries keep their IDs and fitted
feature encoding. Closing a regression removes its owned axes. For one formula across multiple models, provide a
model-qualified option such as `fc.TIRT.loading_magnitude` with a `priors`
mapping to override the TIRT coefficient terms. Other terms inherit the
unqualified options and model defaults. The same selector qualifies `set_priors()`.
New coefficient and target arrays join the prepared design
once per update.

Unit normalization validates the declared model and scale and resolves the
likelihood kind. Response dimensions retain the prepared axes and category indices. Pooled log
likelihood, predictive views, and per-event Pearson discrepancies use the
same prepared `obs` catalogue. TIRTRank's marginal integration uses JAX's native
`ndtri_exp` (JAX 0.12.1 or newer). Only augmented
ranking laws request observed contrast bases and partial-order masks. The
encoder indexes the ranking distribution's geometry once per distinct comparison pattern. Forward
evaluation reads the inputs of the active native response nodes. Posterior
prediction, discrepancies and marginal log likelihood request their outputs
through one native forward evaluator. It reuses the cloned observed RVs and
attaches previously unregistered responses for augmented ranks or schedules.
Prepared batches own the likelihood kinds, event slices and names shared by the
graph, saved outputs and checks. A check with usable cached probabilities reads
those batches and posterior labels without selecting parameter draws or
constructing a model. `fit()` can store ordinary response
log likelihood from retained final draws, with marginal rank likelihood
available via `log_likelihood()`.
Unobserved generation responses
belong to the forward graph rather than the fitting graph. Simulation converts
native draws to normalized response rows using the prepared support. The result
reuses the original administration and attaches the generated responses.
Labelled and indexed administration share the frozen unit records and response join.
Indexed scoring retains the validated entity positions for batch encoding.
An omitted scoring channel contributes administration records only.
Batch encoding projects each observed ranking category once per block, then
gathers its comparisons and source rows in response order. Explicit comparison
inputs retain their prepared local positions and source-direction flag.
Each batch slices the same observation catalogue for person, pair and
administration indices. Log likelihood uses those observation positions to
reuse whole response batches saved for the current posterior. Prediction and
checks select posterior states on one stacked chain–draw axis. Published comparison
provenance uses global statement indices.
TIRT and GTUM share tensor preparation, direction, ordered-cutpoint and
Beta-precision checks in their native response builder. Tensor shapes already
known to PyTensor are retained. The native response and information kernel
use the same bound arguments. The comparison, successive-choice, MGPPM and
single-item models share native response construction. Ordered cutpoints use
one scalar-prior path and PyMC's native batch shape inference. Each model owns
its contrast law.
String and mapping designs share the FC membership table. Cutpoints, pair
regressions and response tables share the prepared pair catalogue. Simulation
reuses the prepared indices and encodes the new observed events through that
same path. Recorded nonresponses keep their status and original pair direction.
The destination publishes its design and newly encoded observations together.
`simulate()` can prepare a balanced forced-choice design and schedule from its
automatic design options. For an existing model, it accepts fixed parameters
as a labelled Dataset or canonical arrays or JSON lists. Preparation labels
these arrays and validates the point before the native graph draws responses.
Generation schedules use the prepared comparison-pair catalogue and pass the
validated indices directly to the encoder. Scoring fills omitted channels after
indexing and reuses unmodified population inputs for the original ordered people.
Scoring, simulation and information share one posterior point reader. It
validates labelled selected variables and applies the same fixed entries.
Data and model summaries share the prepared measurement grouping.
Model and coefficient summaries share canonical fixed entries. `coef()` shows
fixed values under `mean` and leaves uncertainty and diagnostics empty. An explicit
block or item selection reads only its prepared memberships into the coefficient
table. Model summaries read prior roles and population shapes from the prepared layouts.
`Model.attach_samples()` validates finite draws and prepared parameter constraints.
Joint state selection uses the original sample labels. Failed backend selection
with `set_platform()` restores the previous platform and precision settings.
Publication aligns prepared identities on the result's labelled axes. Prior
and regression updates refresh parameter metadata and reuse the frozen
projection of the design and response supports. The model installs one complete
DataTree from preparation, including after local edits to result groups.
Construction consumes each batch's measurement declaration and only the
prepared active slot indices. Graph views reuse registered responses and retained
augmentation nodes, and rebind anonymous expressions together from the final parameters.
The shared response binding applies the prepared
category order to both the likelihood and the derivative kernel. The graph writes free draws directly to canonical
entries. The prepared regression binding supplies pool membership for parameter
usage and retains it when another target is replaced.
The encoder publishes observation-to-source row positions for downstream use.
Ranking formats and partial projections share one read-only complete-order
catalogue per block size within each encoding pass.
TIRTRank prediction and marginal likelihood share one native `pm.logp` expression
per batch in the evaluation graph. Both prediction workflows attach recorded
responses and event support at publication.
Stored and returned-only predictions and likelihoods use the same output
coordinate validation and prepared identity attachment. Cached likelihood factors
stay on `obs` while newly evaluated batches are added.
A new Pearson check uses the same native forward path for replicas and log
probabilities of its selected units.
Prediction views select on the prepared unit and event support carried with the
prediction, including the compared statement pairs for partial rankings.
The population component declares the reference and weighting for marginal
ability correlation; publication attaches that metadata to the result.
Coefficient and trait summaries send selected stochastic values to ArviZ
in one call per summary. Selected fixed trait moments are computed once and can
be reported before fitting.
Coefficient and model summaries share prepared membership indexed by canonical
kind and entry position. Each parameter appears once per measurement unit.
Coefficient summaries use model, group and term labels and show the parameter
set index only when needed to distinguish rows.
