# ItemInput

```{eval-rst}
.. autoclass:: pyfcirt.ItemInput
```

Declare an individual-item design and its recorded answers.

## Parameters

`design` : pandas.DataFrame, str, ItemDesignMapping or mapping
: For a table, provide `item_id` and `trait`, with one row per
  item–trait combination. For a string, list the items for each trait under
  `[traits]`. An item may measure multiple traits.

`responses` : ResponseTable
: In `item_long`, use `person_id`, `item_id` and `response`.
  In `item_wide`, use one row per person and item IDs as answer columns.

`model` : str or measurement specification
: Model for items without a local override. Default:
  `DichotomousIRT(kind="2PL")`. Use `"GPCM"` or `"GGUM"` for ordinal items.

`categories` : tuple of int
: Complete, consecutive integer codes in increasing order, including any
  unobserved categories. Default: `(0, 1)`.

`items` : sequence of ItemOverride or mapping
: Per-item model or category overrides. Default: `()`.

`administration` : pandas.DataFrame or None
: Optional `person_id`, `item_id` and `status` table. The default uses
  the supplied response rows.

## Returns

ItemInput
: A declaration ready for `Model.prepare()` or `PreparedInputs.from_inputs()`.

## Notes

A design table may include `fix_loading` and `location_side`. For GGUM,
declare the location side on every active item–trait combination using +1
(positive), −1 (negative) or 0 (unrestricted).

## Examples

### An ordinal questionnaire


All three items measure theta. Code 3 is a valid answer even if it is absent for an individual item.

```python
import pandas as pd
import pyfcirt as fc

design = pd.DataFrame({"item_id": ["I1", "I2", "I3"], "trait": ["theta"] * 3})
ratings = pd.DataFrame(
    {
        "person_id": ["P1"] * 3 + ["P2"] * 3,
        "item_id": ["I1", "I2", "I3"] * 2,
        "response": [1, 2, 1, 4, 3, 0],
    }
)
answers = fc.ResponseTable(ratings, format="item_long")
inputs = fc.ItemInput(design, answers, model="GPCM", categories=(0, 1, 2, 3, 4))
print(fc.data_coverage(fc.Model().prepare(inputs)).to_string())
```

```text
                       n_items  size  n_persons  n_answered  n_missing
model_response_format
GPCM                         3     1          2           6          0
```

### Mix ordinal and binary items


Use both a binary model and a binary scale for I3.

```python
mixed = fc.ItemInput(
    design,
    answers,
    model="GPCM",
    categories=(0, 1, 2, 3, 4),
    items=(fc.ItemOverride("I3", model="DichotomousIRT", categories=(0, 1)),),
)
print(fc.data_coverage(fc.Model().prepare(mixed)).to_string())
```

```text
                       n_items  size  n_persons  n_answered  n_missing
model_response_format
GPCM                         2     1          2           4          0
DichotomousIRT               1     1          2           2          0
```

## See also

[ResponseTable](pyfcirt.ResponseTable.md) for response layouts and
[Binary and ordinal responses](../../tutorials/binary_ordinal_workflow.ipynb)
for an analysis.
