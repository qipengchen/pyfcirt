# Simulation API requirements

This document describes the target API. The published simulation reference
describes the behavior available in the current release.

## `simulate()`

- `fc.simulate()` generates an FC design, a complete administration schedule
  and one response dataset from the model's existing priors. The defaults are
  100 respondents, 3 traits, 12 triplets, `MGPPM` and `rank`. The automatic route
  accepts optional `model`, `n_persons`, `n_traits`, `n_blocks`, `block_size`,
  `response_format` and `random_seed`. `model` may be a supported FC model name
  or declaration. With a joint block model, the response format defaults to
  `rank`. Categorical TIRT and GTUM default to binary comparison judgments.
  Use an explicit design and scale for graded or continuous comparisons.
- With a prepared `Model`, use `model.simulate(...)` or
  `fc.simulate(model, ...)`. A custom design still enters through
  `FCInput` or `ItemInput` and `Model.prepare(..., purpose="generate")`, with a
  schedule that has no answers. A prepared model supplies the design and size,
  so automatic design options apply only when generating a new design.
- `source="prior"` is the default. With a prepared model,
  `source="fixed"` accepts a labelled `xarray.Dataset` or a mapping from
  canonical parameter names to numeric arrays or lists. JSON input is a mapping
  read with `json.load()`. With `source="posterior"`, a fitted or restored model
  supplies one joint draw. Optionally select it with
  `draw=(chain_label, draw_label)`.
- All routes return an independent, unfitted `Model`. Its `idata["design"]`
  records the resolved design, and `idata["parameters/generation"]` records the
  parameter values that generated the responses. `fc.response_data(model)`
  displays the categorical responses, including comparison judgments.

Data preparation creates one statement per block position and assigns distinct
traits within a block. Choose traits with the fewest past uses, break ties
randomly and shuffle their positions. Assign stable IDs and construct the
ordinary design rows with `block_id`, `position`, `statement_id`, `trait`
and `keying`. Automatic designs need positive counts, at least two positions
per block, at least as many traits as block positions, and enough slots to
include every trait. Use the existing model declarations and preparation checks for
required design fields and identification, including loading references and
ideal-point location sides where applicable. Every respondent receives the same
generated design. A supplied seed controls both design and response generation,
with separate random streams for each. This design rule is adapted from
[thurstonianIRT](https://github.com/paul-buerkner/thurstonianIRT/blob/master/R/simulations.R#L230).

For mapping input, data preparation assigns axes and coordinates from the
prepared layouts and checks every array's expected shape. The names and
positions are those of the final parameters, including active shared entries.
Fixed entries included in an array must agree with the prepared values. Report
unknown names, missing required parameters, shape mismatches and violations of
prepared constraints. Use the same validation as labelled points. With the
original administration, supply `theta` and any required individual effects.
With a new administration, supply the global calibration values and draw new
abilities and individual effects from the existing population model.

## `generation_schedule()`

Use `model.generation_schedule(person_ids, person_data=None)` or
`fc.generation_schedule(model, person_ids, person_data=None)` to apply an
existing design to new IDs. Provide `person_data` when the model uses person
predictors or groups. The function returns a labelled `xarray.Dataset` to pass
as `administration=` to `simulate()`.

## Example 1. Generate a design and responses without input

```python
import pyfcirt as fc

simulated = fc.simulate()
design = simulated.idata["design"].to_dataset()
print(design[["fc_statement", "statement_label", "statement_trait"]])
print(fc.response_data(simulated).head())
```

`fc_statement` gives the statements in each block's presentation order.
`statement_label` and `statement_trait` connect those entries to the statement
IDs and the trait axis. The default design yields 1,200 complete rankings.
Pass `random_seed=42` to reproduce the design and responses.

## Example 2. Set all generating parameters for a TIRT triplet

```python
import json

import pandas as pd
import pyfcirt as fc

design = """
[traits]
agency ~ A C
communion ~ B
[blocks]
B1 ~ A B C
"""
schedule = pd.DataFrame(
    {
        "person_id": ["P1"] * 3 + ["P2"] * 3,
        "block_id": "B1",
        "statement_1": ["A", "A", "B"] * 2,
        "statement_2": ["B", "C", "C"] * 2,
    }
)
inputs = fc.FCInput(
    design,
    fc.ResponseTable(schedule, format="comparison_schedule"),
    model=fc.TIRT(residuals="fixed", loading_anchors={"A": 1.0, "B": 1.0}),
    response_format=None,
)
model = fc.Model().prepare(inputs, purpose="generate")

with open("parameters.json", encoding="utf-8") as stream:
    parameters = json.load(stream)

simulated = model.simulate(source="fixed", parameters=parameters, random_seed=42)
print(fc.response_data(simulated)[["person", "pair_label", "category"]])
```

`parameters.json` contains the complete parameter point for this example.
The same structure can be written in Python using NumPy arrays.

```json
{
  "loading": [1.0, 1.0, 1.2],
  "statement_effect": [0.1, 0.0, -0.1],
  "residual_variance": [0.5, 0.5, 0.5],
  "cutpoint": [0.1, -0.2, 0.3],
  "population_mean": [[0.0, 0.0]],
  "covariance": [[1.0, 0.25], [0.25, 1.0]],
  "theta": [[0.4, -0.3], [-0.2, 0.6]]
}
```

To use NumPy arrays, pass
`{name: np.asarray(values) for name, values in parameters.items()}` as
`parameters` after importing `numpy as np`.

- `loading` gives the statement's trait slope before keying. The order is
  `A (agency)`, `B (communion)`, `C (agency)`. The first two are fixed
  loading anchors.
- `statement_effect` gives each statement's utility offset in the same order.
- `residual_variance` gives each statement's residual variance in the same
  order. The model fixes all three at 0.5.
- `cutpoint` gives the unstandardized binary threshold for `A vs B`,
  `A vs C`, and `B vs C` in `B1`, in that order. A higher threshold shifts
  probability toward the second statement under the default answer coding.
- `population_mean` has one row for the default group. Its columns are
  `agency` and `communion`. Both means are fixed at zero for this design.
- `covariance` contains the population trait covariance. Its rows and columns
  are both ordered `agency`, `communion`. The diagonal is fixed at one here.
- `theta` contains the abilities used for the six comparisons. Rows are
  `P1`, `P2`. Columns are `agency`, `communion`.

These orders belong to this prepared design. In the implemented reference,
show how to inspect the prepared parameter coordinates and map entries to
statements, traits or pairs before filling arrays for a different design.

## Implementation and acceptance

Data preparation owns generated design rows and the mapping-to-labelled-point
conversion. Simulation owns orchestration, then uses existing preparation,
model graphs and generation truth publication. Update the full and concise API
references before changing source behavior.

Run both examples end to end with deterministic regression assertions. Check
seed reproducibility, balanced trait use and distinct traits per block, valid
observed response encoding, and exact truth and coordinates for the array and
JSON inputs. Cover bad keys, missing required arrays, wrong shapes, conflicting
fixed or shared entries, and new-person scheduling. Check that the existing
labelled-input and posterior routes still work. Then run the full local suite
once. Investigate failures, fix any confirmed problems according to the six
principles in `AGENTS.md` and review the resulting changes. Keep work local.
