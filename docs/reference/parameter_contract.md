# Parameter and coordinate contract

A prepared design assigns identities to parameters and responses before the
model is built. Saved results keep those identities so that xarray and ArviZ
can select samples by their dimensions, and PyFCIRT can associate them with
blocks, items, traits and response categories.

## Parameter arrays

Within `model.idata.posterior`, values use the same names in forced-choice
and item models whenever they have the same interpretation:

| Variable | Dimensions after `chain`, `draw` | Interpretation |
| --- | --- | --- |
| `loading` | `loading_id` | One entry per estimated, shared or fixed loading. Effective utility slopes also apply the declared keying where the model uses it. |
| `cutpoint` | `cutpoint_id` | One entry per pair cutpoint and cutpoint position. |
| `location` | `location_id` | One entry per active ideal point. |
| `regression_coefficient` | `regression_coefficient_id` | Entries link to the target, formula term and any applicable trait or group. |
| `theta` | `person`, `trait` | Person trait values. |
| `population_mean` | `population_group`, `trait` | Population trait means. |
| `trait_correlation` | `trait`, `trait_aux`; also `population_group` if group-specific | Residual trait correlation. Grouped models can opt into one shared matrix with `theta_correlation="shared"`. |

Only quantities applicable to the model are present. Entirely fixed quantities
can be absent from the posterior; consult `parameter_data/<kind>`. Sampled
vectors may include a mixture of fixed and estimated entries.

An entry ID is the label on a kind's axis, for example `loading_id`. An
auxiliary coordinate such as `loading_item_index` holds integer **positions**
into the saved `item` catalogue. `-1` denotes an inapplicable relationship
and is never a valid selection position. Other coordinates include applicable
trait, statement, pair, cutpoint position, parameter set and regression/term
positions. A relationship inapplicable to every entry is omitted. Xarray
auxiliary coordinates can be filtered with `where(..., drop=True)`.

Use [coef()](generated/pyfcirt.coef.md) to select complete blocks, items or
parameters. It applies the `parameter_uses` relationship so that a shared
statement or the common 1PL loading has one posterior entry and can be displayed
wherever it is used. Fixed entries display the declared value with blank
uncertainty and diagnostics.

## Result groups and relationships

| Group | What it identifies |
| --- | --- |
| `design` | Typed entity IDs and the unit, block membership, pair and `obs` catalogues. A unit is a block or an item. A pair belongs to a block and has two ordered statement positions. |
| `categories` | Complete declared scalar supports, including unused categories, in unit/position order. Integer and text representations retain their native meaning. |
| `response_support/<batch>` | Batch-local category patterns or comparison signatures and the associated native observation dimension. Unknown ranking positions use `-1`. |
| `parameter_data/<kind>` | Entry identities, fixed values and their sources, domains, joint constraints and reporting metadata. `NaN` in a fixed-value field means no fixed constant. |
| `parameter_uses` | One-to-many uses linking entries to units and within-unit positions. |
| `fixed_parameters` | The fixed calibration point for scoring or simulation, when present. |
| `calibration_evidence` | For recursive scoring, incoming evidence IDs in `people(evidence_person)` and `batches(evidence_batch)`. |

Primary IDs use their declared integer or string type. A prepared relationship
position refers to the corresponding catalogue in its original order. Different
catalogues can use different ID types. Parameter-set and regression IDs
are assigned at preparation. Surviving IDs remain stable when another
regression pool is removed, so the axis can have gaps. Keep the xarray labels when selecting
posterior draws or supplying final parameter points. Auxiliary projection uses
the same identities as preparation. It rejects an existing conflicting
coordinate.

### Observation units

`log_likelihood.response(chain, draw, obs)` has one entry per likelihood
factor, ordered by prepared `obs` ID. It refers back to respondents, units and
pairs through observation coordinates such as `obs_person_index`,
`obs_unit_index` and `obs_pair_index`. The source-row relationship preserves
the direction in which comparison answers were supplied. One recorded ranking
can contribute several factors in TIRT or GTUM. Joint ranking models use one
factor for its event. TIRTRank's stored response log likelihood integrates
its fitting auxiliaries out over the observed preference region.

The possible answers to the original task and the outcome support of a
likelihood factor are separate. For a triplet the six complete ranks and
the two outcomes of each TIRT binary comparison give different counts.
`data_coverage(details=True)` reports both. Prediction keeps the categories of
each native batch together. Use [predictive_view()](generated/pyfcirt.predictive_view.md)
for matching observed and replicated values. For a block with multiple response
supports, read the native batch arrays using `response_support` instead.

### Constraint and prior binding

Preparation resolves statistical declarations against the actual entities.
Inactive item–trait cells have no loading entry. Shared entries have one
construction rule and can have several uses. Fixed entry values, ordered groups
and dependent bounds are explicit. For example, a 4PL upper asymptote links to
the corresponding lower asymptote. The native distribution uses that lower
bound along with its own support. External parameter values are checked against
the same fixed and joint constraints. Fixed-value validation allows absolute
roundoff of eight machine epsilons at the supplied floating-point precision
with zero relative tolerance. Fixed integer values must match exactly.

Different prior laws use the prepared entries assigned to each measurement
role. Use a model-qualified selector such as
`fc.TIRT.loading_magnitude` to select one family. Existing joint priors and
ordered cutpoint distributions retain their joint law. A regression target uses
its fitted predictor rows, coefficient pool and prepared target entries.
Known predictor encodings carry over to new respondents. Measurement parameters
normally remain shared across population groups; group-specific trait
distributions retain their own labelled axes.

Changing a specification creates a new prepared candidate. This resolves
prior and regression bindings before the graph changes. Fixed identification
values and unaffected parameter relationships carry through. Sampled results
require the specification under which they were obtained.

## Example: find the input IDs behind a coefficient

The example loads an item design with three respondents and prepares it
without sampling. The integer values in the loading identity coordinate select
the corresponding item labels.

```python
import pyfcirt as fc

bundle = fc.load_example("simulated_mirt", model="DichotomousIRT", n_persons=3)
prepared = fc.PreparedInputs.from_inputs(bundle.data)
item_positions = prepared.output_coords.loading_item_index.values[:3]
print(prepared.constant_data.item.values[item_positions].tolist())
```

```text
['I1', 'I2', 'I3']
```

## Save and reuse

Save the result tree using xarray. Recreate the model from its original data
and model declarations, then use [attach_samples()](generated/pyfcirt.Model.attach_samples.md).
The result contains metadata and samples, while native priors, fitted formula
encoders and a live PyMC graph remain in memory. Attachment checks the supplied
sample axes, parameter identities, constraints and observation batch.

For recursive scoring, preserve `calibration_evidence` along with the
posterior. It records the respondents and batches covered by the incoming
approximation. Scoring checks new respondents against the verified history.
Reconstructing the round for fitting or prior prediction also requires its
original runtime approximation.

See [Saving and reusing results](results.md) for the full workflow and
[data preparation](data.md) for the originating records.

## Respondent-level measurement regressions

A measurement parameter keeps its existing canonical entry and fixed-value
policy. The ordinary prior or measurement-level regression supplies its
baseline. Respondent-level terms supply deviations on the target's link scale.
The prepared binding identifies the measurement entry, the coefficient pool
for that entry, and the aligned respondent design row for each observation.
Fixed entries keep the declared value and have no respondent effect. The
measurement family continues to own parameter constraints and response laws.

The preparation layer resolves targets through the existing measurement
relations and fits the respondent formula through Formulae. A frozen encoder
and coefficient identities survive a change in respondent cohort. Measurement-
and respondent-level declarations update their respective components
atomically. The graph gathers their values for the requested observations.
It saves canonical baselines and coefficient draws without materializing a
respondent-by-measurement posterior array.

An ordinary categorical term is zero at its encoded reference. A PairwiseL1
term uses a sum-to-zero group basis and one joint contrast event per pool.
Each group, including the treatment reference, has an offset on that basis.
One pool may serve multiple free parameters when sharing is explicit. An
ordered cutpoint vector receives a common shift. Positive scales and signed
half-axis locations receive multiplicative effects via the log link.

The fitted-unit coefficients are obtained from the saved coefficient draws
and any retained hierarchical scales using their prepared pool indices.
Explicit coefficient priors and joint contrast priors use fitted units
directly. The analysis uses the frozen encoder to compare two profiles at
the same baseline draw, then reports the difference on the parameter scale.
See [differential item functioning](dif.md) for the public methods.
