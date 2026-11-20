# Model

```{eval-rst}
.. autoclass:: pyfcirt.Model
```

Create a model from your design and responses. Start with `fc.Model()` to add
your own tables, or pass an existing data declaration to prepare it immediately.

## Parameters

`prepared` : PreparedInputs, FCInput, ItemInput, tuple of inputs or None
: An existing prepared design or data declaration. A tuple can combine
  forced-choice and individual-item declarations. With `None`, register your
  tables and call `prepare()`.

`marginal_correlation` : bool or None
: Include the marginal trait correlation across the reference respondents
  when their trait scores have predictors. `None` uses the prepared setting.
  For an empty model, pass this option to `prepare()`.

## Returns

Model
: `model.prepared` holds the validated design after preparation. `model.idata`
  contains design metadata and, after fitting, posterior draws. Use
  [`build()`](pyfcirt.Model.build.md) to obtain the native PyMC model.

## Prepare data

```{eval-rst}
.. automethod:: pyfcirt.Model.prepare
```

After using `add_fc()` or `add_items()`, call `model.prepare()` without a
data argument. You can register both sets of responses before one
`prepare()` call to analyze them jointly. If you already have a data
declaration, pass it directly instead. Options such as `person_data`
and `population` go to `prepare()`. See
[preparation options](pyfcirt.PreparedInputs.from_inputs.md).

For example, prepare the design and answers from a bundled dataset.
This returns the prepared `Model`. Call `data_summary()` to check the
numbers of respondents and recorded answers.

```python
import pyfcirt as fc

example = fc.load_example("tirt_study", model="TIRT", n_persons=3)
model = fc.Model().prepare(example.data)
print(model.data_summary()[["n_persons", "n_answered"]].to_string())
```

```text
                       n_persons  n_answered
model_response_format
TIRT                           3          72
```

### Add a forced-choice response table

```{eval-rst}
.. automethod:: pyfcirt.Model.add_fc
```

Pass the design and the response DataFrame directly to `add_fc()`.
In a design DataFrame, each row gives the `block_id`, the 1-based
`position`, the `statement_id`, the measured `trait` and the `keying`
sign (`1` or `-1`). In the response table below, `[3, 1, 2]` ranks
the statements at positions 3, 1 and 2, so it means C > A > B.

```python
import pandas as pd
import pyfcirt as fc

design = pd.DataFrame(
    [("B1", 1, "A", "agency", 1), ("B1", 2, "B", "social", 1), ("B1", 3, "C", "agency", -1)],
    columns=["block_id", "position", "statement_id", "trait", "keying"],
)
responses = pd.DataFrame(
    [("P1", "B1", [3, 1, 2]), ("P2", "B1", [2, 3, 1])],
    columns=["person_id", "block_id", "response"],
)

model = fc.Model()
model.add_fc(
    design,
    responses,
    format="block_long",
    model="MGPPM",
    response_format="rank",
)
model.prepare()
print(model.data_summary()[["n_blocks", "n_answered"]].to_string())
```

```text
                       n_blocks  n_answered
model_response_format
MGPPM / rank                  1           2
```

`format` describes table storage. Use `"block_long"` when each row gives
one block answer, or `"comparison_long"` when each row gives a pair
and includes `statement_1` and `statement_2`. `response_format`
describes the task. Set it to `"rank"` for a full ranking, `"pick"`
for the top choice, `"mole"` for the top and bottom choices, or `None`
for separate pair judgments. Comparisons derived from a ranking keep
`response_format="rank"`. For wide tables, see the
[response layout reference](pyfcirt.ResponseTable.md).

#### Use a design string

You can replace the DataFrame in the same call with this string.
`[traits]` assigns statements to traits and `[blocks]` gives their
presentation order. `sign -` marks negative keying. The other statements
are positively keyed by default.

```python
design_string = """
[traits]
agency ~ A C
social ~ B
sign - ~ C

[blocks]
B1 ~ A B C
"""

model = fc.Model()
model.add_fc(
    design_string,
    responses,
    format="block_long",
    model="MGPPM",
    response_format="rank",
)
model.prepare()
```

#### Set location sides for ideal-point models

For `"GGUMRank"` or `"GTUM"`, declare the side of each statement's
ideal-point location. In a DataFrame, add a `location_side` column. Set
`1` for a positive location, `-1` for a negative location, or `0` for
an unrestricted location. These signs constrain the location on the
trait scale. The `keying` column describes the direction of a dominance
loading.

For the three statements above, use
`design.assign(location_side=[1, -1, 0])`. With the string design,
replace the `sign - ~ C` line with these directives in `[traits]`:

```text
side + ~ A
side - ~ B
side 0 ~ C
```

The ranked answers above can then be used with `model="GGUMRank"`.
[GTUM](pyfcirt.GTUM.md) uses pairwise choice or preference responses.

### Add an individual-item response table

```{eval-rst}
.. automethod:: pyfcirt.Model.add_items
```

Pass a design table with `item_id` and `trait`, with one row for each
item and trait it measures. For `"item_long"`, provide `person_id`,
`item_id` and `response`. For `"item_wide"`, give one row per person
with a column named for each item. Use `person_id_column` if the
respondent ID is a column in the wide table.

This example adds two ordinal items. Declare the whole response scale,
including categories that no one selected in the example.

```python
import pandas as pd
import pyfcirt as fc

items = pd.DataFrame({"item_id": ["I1", "I2"], "trait": ["agency", "agency"]})
ratings = pd.DataFrame(
    [("P1", "I1", 2), ("P1", "I2", 0), ("P2", "I1", 1), ("P2", "I2", 2)],
    columns=["person_id", "item_id", "response"],
)

model = fc.Model()
model.add_items(
    items,
    ratings,
    format="item_long",
    model="GPCM",
    categories=(0, 1, 2, 3),
)
model.prepare()
print(model.data_summary()[["n_items", "n_answered"]].to_string())
```

```text
                       n_items  n_answered
model_response_format
GPCM                         2           4
```

For binary items use `model="DichotomousIRT"` and `categories=(0, 1)`.
For [GGUM](pyfcirt.GGUM.md), add `location_side` to each item–trait row.

## Notes

The add methods return the model. Calling one again before `prepare()`
replaces the registered tables for that channel. Create a fresh `Model`
to prepare a different dataset.

## See also

[Reusable forced-choice inputs](pyfcirt.FCInput.md),
[reusable individual-item inputs](pyfcirt.ItemInput.md),
[fit a model](pyfcirt.Model.fit.md) and
[score new respondents](../scoring.md).
