# ResponseTable

```{eval-rst}
.. autoclass:: pyfcirt.ResponseTable
```

Declare the storage format of observed responses or a generation schedule.

## Parameters


`frame` : pandas.DataFrame
: Recorded answers, or a schedule with person and unit IDs but no answers.

`format` : str
: Storage layout. Choose one of the formats below.

`person_id_column` : str, int or None
: For a wide table, the column holding respondent IDs. `None` uses the index.

`comparison_columns` : tuple of ComparisonColumn
: For `comparison_wide`, exactly one descriptor per answer column. Default: `()`.


## Returns

ResponseTable
: A table declaration to pass to `FCInput` or `ItemInput`. Preparation
  validates the declared layout against the design.

## Notes


### Response format and storage layout

`format` describes storage. `FCInput.response_format` describes the task,
even when answers are stored as comparisons.

| Storage format | Required columns or layout |
| --- | --- |
| `block_long` | `person_id`, `block_id`, `response` |
| `statement_rank` | `person_id`, `block_id`, `statement_id`, `rank` |
| `comparison_long` | `person_id`, `block_id`, `statement_1`, `statement_2`, `response` |
| `block_wide` | One row per person, with block IDs as answer-column names |
| `comparison_wide` | One row per person, with explicitly mapped pair columns |
| `item_long` | `person_id`, `item_id`, `response` |
| `item_wide` | One row per person, with item IDs as answer-column names |

Use the design's exact ID labels and types. In long tables, key each answer
by person and unit, plus statement or pair when applicable. Wide tables
require one row per person. Use an empty cell for a missing answer. An
optional `status` column on long rows can distinguish `answered`,
`missing`, `refused` and `not_administered`.

With `purpose="generate"`, use `block_schedule`, `comparison_schedule`
or `item_schedule`. Include the identifying columns of the corresponding
long layout and omit the response column.

### Categories and comparison direction

An order lists one-based presentation positions from most to least preferred.
For A, B, C, D, RANK is e.g. `[3, 2, 4, 1]`, PICK is
`[3, None, None, None]` and MOLE is `[3, None, None, 1]`. In JSON
cells use `null` in place of `None`. In `statement_rank`, C has rank 1
and A has rank 4 in the MOLE example.

For independent binary pair judgments the default is 1 for the first
recorded statement and 0 for the second. Declare a `CategoryScale` for
graded scores or a `UnitIntervalScale` for continuous scores. A derived
pair from a ranking keeps the original RANK, PICK or MOLE task.

When adding a DataFrame through [`Model.add_fc()`](pyfcirt.Model.md#add-a-forced-choice-response-table)
or [`Model.add_items()`](pyfcirt.Model.md#add-an-individual-item-response-table), pass the
storage format to that method. Create a `ResponseTable` to pass the responses separately.

## Examples

### Ranking responses in a CSV


This inline CSV is equivalent to reading a file with pandas. The row rates C above A above B for presentation order A, B, C.

```python
from io import StringIO
import pandas as pd
import pyfcirt as fc

csv = 'person_id,block_id,response\nP1,B1,"[3, 1, 2]"'
rows = pd.read_csv(StringIO(csv))
answers = fc.ResponseTable(rows, format="block_long")
design = {"traits": {"a": ["A"], "b": ["B"], "c": ["C"]}, "blocks": {"B1": ["A", "B", "C"]}}
model = fc.Model().prepare(fc.FCInput(design, answers))
print(fc.data_summary(model).to_string())
```

```text
                       n_blocks  size  n_persons  n_answered  n_missing
model_response_format
MGPPM / rank                  1     3          1           1          0
```

### A wide pair table


Column choice records the preference between A and B. Here one respondent prefers each statement.

```python
choices = pd.DataFrame({"person_id": ["P1", "P2"], "choice": [1, 0]})
answers = fc.ResponseTable(
    choices,
    format="comparison_wide",
    person_id_column="person_id",
    comparison_columns=(fc.ComparisonColumn("choice", "B1", "A", "B"),),
)
dyad = {"traits": {"theta": ["A", "B"]}, "blocks": {"B1": ["A", "B"]}}
model = fc.Model().prepare(
    fc.FCInput(dyad, answers, model=fc.TIRT(loading_anchors={"A": 1}), response_format=None)
)
print(fc.data_summary(model).to_string())
```

```text
                       n_blocks  size  n_persons  n_answered  n_missing
model_response_format
TIRT                          1     2          2           2          0
```

## See also

[FCInput](pyfcirt.FCInput.md), [ItemInput](pyfcirt.ItemInput.md),
[ComparisonColumn](pyfcirt.ComparisonColumn.md) and [study files](../../datasets.md).
