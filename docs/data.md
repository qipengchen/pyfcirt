# Working with forced-choice data

`FCDesign` and `FCData` are implemented. They validate and preserve data without
building a PyMC graph. See the [model guide](model.md) for the fitting interface.
Use `import pyfcirt as fc` and keep one response row per person and block.

## Define statements, traits and blocks

```python
import pandas as pd
import pyfcirt as fc

design = fc.FCDesign.from_mapping(
    traits={"extraversion": ["S1"], "agreeableness": ["S2"], "openness": ["S3"]},
    blocks={"B1": ["S1", "S2", "S3"]},
    keying={"S2": -1},
)
responses = pd.DataFrame(
    {
        "person_id": ["P1"],
        "block_id": ["B1"],
        "response": [[2, 3, 1]],
    }
)
data = fc.FCData.from_frames(responses=responses, design=design)
print(data.to_rank_frame())
```

The list `[2, 3, 1]` means position 2 is preferred to position 3, which is
preferred to position 1. Statement ranks are therefore `[3, 1, 2]` in reference
position order. Do not interchange these encodings. Negative keying describes
trait direction and never reverses the observed ranking automatically.

The table alternative is `FCDesign.from_frame(frame)`, with required columns
`block_id`, `position`, `statement_id`, `trait`, and `keying`. Position is
one-based; keying is +1 or -1. A statement has one trait in this release. If a
statement occurs in multiple blocks, its trait, keying and optional text must
agree. Data-level acceptance does not establish model identification.

`design.to_frame()`, `design.statements`, `design.summary()` and
`design.block_sizes` expose independent views. A uniform design has an integer
`block_size`; a variable-size design has `block_size is None`. Enable different
sizes explicitly with `allow_variable_block_size=True`. No rows are padded to
the maximum size. Optional statement features are stored by statement ID;
storing features does not yet implement item-parameter regression.

IDs are nonempty strings or integers. Position and rank values must be exact
integers; boolean values are not valid positions. Public table views and nested
response lists are copied, so mutating an input or exported view does not
silently change the stored data.

## Full rankings, PICK and MOLE

Set `response_format="rank"`, `"pick"` or `"mole"` on the design. The
`block_response_formats` mapping can override the default for particular blocks.
The task is declared, not inferred from the NA pattern.

- Full ranking: `[2, 3, 1]` for a triplet.
- PICK: `[2, None, None]` for a triplet.
- MOLE: `[2, None, None, 4]` for a tetrad.

Every list has exactly the block's size. NA occupies unknown rank slots; it
does not delete statements from the choice set. A triplet MOLE response still
retains its middle NA even when the remaining position is logically determined.
The later likelihood adapter sums compatible complete-ranking probabilities.

Python lists, tuples and one-dimensional NumPy arrays are accepted. CSV response
cells must contain JSON arrays such as `[2, null, null, 4]`. Compact digit strings
and executable Python strings are not accepted or evaluated.

For legacy long data, supply `encoding="statement_rank"` and columns
`person_id`, `block_id`, `statement_id`, `rank`. Include every statement row of
each observed block, including structural NA rows. `rank=1` means most preferred;
zero is not missing. Sparse selected-statement rows are rejected. Export the
ranking view with `data.to_rank_frame()`; its rank column uses nullable integers.

## Graded pairs

A graded block has exactly two statements and one scalar category response.
Declare the complete scale even if some categories do not occur in the sample:

```python
pair_design = fc.FCDesign.from_mapping(
    traits={"T1": ["A"], "T2": ["B"]},
    blocks={"pair": ["A", "B"]},
    response_format="graded",
    graded_scale={
        "categories": [1, 2, 3, 4, 5],
        "higher_prefers": "position_1",
        "labels": ["strong_second", "second", "equal", "first", "strong_first"],
    },
)
pair_responses = pd.DataFrame(
    {
        "person_id": ["P1"],
        "block_id": ["pair"],
        "response": [5],
    }
)
pair_data = fc.FCData.from_frames(pair_responses, pair_design)
```

Category order comes from `categories`, not lexical or numeric sorting.
`higher_prefers` is explicitly `position_1` or `position_2`. Labels are optional
and must have the same length as categories. A neutral response is an ordinary
declared category, not missing data.

A design can contain ranking and graded tasks. `data.to_frame()` preserves both
in the default block-event representation. `to_rank_frame()` returns answered
ranking tasks only; `to_graded_frame()` returns answered graded events only.
The latter are never duplicated as statement ranks. The explicit statement-rank
input encoding cannot represent graded events.

## Genuine missingness and administration

For an answered PICK/MOLE event, structural NA is already part of the task.
An entirely missing response is different and requires explicit administration
information. Supply `administration` with `person_id`, `block_id` and `status`:

- `answered` requires one valid response.
- `missing` has no effective event and must not contain known choices.
- `not_administered` also has no effective event and must not contain choices.

When provided, administration must include every response pair, with unique
person/block entries and known blocks. Missing/unadministered response rows may
be absent or contain an entirely missing response. People with no effective
responses are retained through their administration entries. The package does
not guess whether absent rows represent omitted or unadministered blocks.

`data.person_ids` includes those people. `data.to_frame(answered_only=True)`
returns effective events, while `data.administration` retains statuses.
`data.summary()` counts one observation per answered person/block, not one per
statement or pairwise comparison. Ignoring genuinely missing events does not
model a nonignorable missing-data mechanism.

## Reusing the shared tutorial inputs

`scripts/prepare_tutorial.py` explicitly converts zero-based tutorial positions
and category codes. It validates the converted data, retains source checksums,
keeps respondent slots from the person table and writes JSON ranking arrays.
Run it with `--input`, `--output` and `--task rank` or `--task graded`.
No input data is bundled or downloaded by the script. The shared ranking example
is empirical; the graded example is simulated. Conversion does not establish a
redistribution license. See the [MGPPM specification](models/mgppm.md) for the
statistical target and provenance distinctions.


## Binary and multi-category responses

The same three response columns are used throughout: `person_id`, `block_id`,
`response`. Their values depend on the declared task:

- **Binary pair preference:** use `response_format="rank"`, two statements per
  block, and lists `[1, 2]` or `[2, 1]`. Raw 0/1 values are not rank input; map
  them explicitly according to which statement the original code selected.
- **Full ranking of three statements:** use lists such as `[2, 3, 1]`. This is
  one of six possible ranking events, not a scalar graded-response score.
- **Graded pair preference:** use `response_format="graded"`, two statements per
  block, a declared `graded_scale`, and one scalar original category per event.
  Current MGPPM supports five-category graded pairs; a two-category graded model
  or arbitrary numbers of graded categories are not implemented.

For example, declare `categories=[1, 2, 3, 4, 5]` and
`higher_prefers="position_1"`; then response `5` favors the first statement more
than response `4`. Keying still describes trait direction, not response recoding.
The input data layer accepting a scale does not establish model support for it.
