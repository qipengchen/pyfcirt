# FCInput

```{eval-rst}
.. autoclass:: pyfcirt.FCInput
```

Declare a forced-choice design and its recorded answers.

## Parameters

`design` : pandas.DataFrame, str, FCDesignMapping or mapping
: For a DataFrame, provide `block_id`, `position` (starting at 1 within each
  block), `statement_id`, `trait` and `keying` (+1 or −1). For a string,
  use `[traits]` and `[blocks]` as below. A shared statement ID denotes the
  same statement wherever its measurement specifications are compatible.

`responses` : ResponseTable
: Answers in a declared [storage layout](pyfcirt.ResponseTable.md).

`model` : str or measurement specification
: Model for blocks without a local model. The default is `MGPPM()`.

`response_format` : {"rank", "pick", "mole"} or None
: What respondents record. Use `"rank"` for a complete order, `"pick"` for
  the top choice, `"mole"` for the top and bottom choices, or `None` for
  separate pair judgments. The default is `"rank"`.

`scale` : CategoryScale, UnitIntervalScale or None
: Preference scale for pair judgments. `None` uses binary 0/1 (1 favors the
  first statement) for separate pairs, or the declared event categories for
  ranking tasks. Declare ordinal and continuous pair scales explicitly.

`blocks` : sequence of BlockSpec or mapping
: Per-block overrides. The default `()` leaves local design settings in place.

`administration` : pandas.DataFrame or None
: Optional table with `person_id`, `block_id` and `status`. Otherwise
  administration is inferred from supplied response rows.

`statement_features` : pandas.DataFrame or None
: Optional feature table with one row per `statement_id`.

## Returns

FCInput
: A declaration ready for `Model.prepare()` or `PreparedInputs.from_inputs()`.

## Notes

The same field resolves in this order: `blocks=` overrides, local design
settings, wildcard settings in the design string, then input defaults. For
example, `B2 ~ D E F [TwoPLRank, pick]` declares a local model and response
format. A table can use the columns `model`, `response_format` and `scale`
for local options.

Optional design columns are `location_side` (+1, −1 or unrestricted 0),
`fix_loading` (magnitude) and `fix_var` (residual variance). Empty fixed
values are estimated subject to the model's identifying constraints. In a
design string, `sign - ~ C F` sets negative keying and `side + ~ A D` sets
positive ideal-point sides. See [loading anchors](pyfcirt.Model.set_priors.md#tirt-loading-anchors).

## Examples

### Complete designs


The answers list presentation positions from most to least preferred. In B1, [3, 1, 2] means C > A > B.

```python
import pandas as pd
import pyfcirt as fc

design = """
[traits]
agency ~ A D
social ~ B E
stability ~ C F
[blocks]
B1 ~ A B C
B2 ~ D E F
"""
rows = pd.DataFrame(
    {
        "person_id": ["P1", "P1", "P2", "P2"],
        "block_id": ["B1", "B2", "B1", "B2"],
        "response": [[3, 1, 2], [1, 3, 2], [2, 1, 3], [3, 2, 1]],
    }
)
inputs = fc.FCInput(design, fc.ResponseTable(rows, format="block_long"))
print(fc.data_summary(fc.Model().prepare(inputs)).to_string())
```

```text
                       n_blocks  size  n_persons  n_answered  n_missing
model_response_format
MGPPM / rank                  2     3          2           4          0
```

### Give one block a different model


B2 still uses complete rankings. Only its measurement model changes.

```python
mixed = fc.FCInput(
    design,
    fc.ResponseTable(rows, format="block_long"),
    blocks=(fc.BlockSpec("B2", model="TwoPLRank"),),
)
print(fc.data_summary(fc.Model().prepare(mixed)).to_string())
```

```text
                       n_blocks  size  n_persons  n_answered  n_missing
model_response_format
MGPPM / rank                  1     3          2           2          0
TwoPLRank / rank              1     3          2           2          0
```

## See also

[ItemInput](pyfcirt.ItemInput.md) for binary and ordinal items;
[prepared inputs](pyfcirt.PreparedInputs.from_inputs.md) to combine channels.
