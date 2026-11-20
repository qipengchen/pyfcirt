# response_data

```{eval-rst}
.. autofunction:: pyfcirt.response_data
```

Get labelled observed answers for descriptive analysis.

## Parameters

`data` : Model or PreparedInputs
: Recorded answers and design. No posterior is required.

`blocks`, `items` : sequence or None
: Block or item IDs. Omit both to include all units.

`representation` : str or mapping
: `"auto"` uses pairwise rows for TIRT/GTUM and category rows for other
  models. Specify `"pairwise"` or `"categories"`, or map individual block
  IDs to these choices. Items use categories.

`groups` : pandas.DataFrame or None
: Group columns indexed by person ID, with index named `person`.

## Returns

pandas.DataFrame
: Long table with `person`, `block` or `item`, and the observed
  `category` plus `category_label`. A pairwise view includes the typed
  `pair` and its preference direction in `pair_label`. Only answered rows appear.

## Notes

A pair label A > B means that a larger response favors A. The category column gives the recorded answer.

## Examples

Inspect the first three pair records in one triplet.

```python
import pyfcirt as fc

bundle = fc.load_example("tirt_study", model="TIRT", n_persons=6)
model = fc.Model(bundle.data)

answers = model.response_data(blocks=["B1"])
columns = ["person", "block", "pair", "pair_label", "category"]
print(answers[columns].head(3).to_string(index=False))
```

```text
person block     pair pair_label  category
    P1    B1 (S1, S2)    S1 > S2         0
    P6    B1 (S1, S2)    S1 > S2         0
    P2    B1 (S1, S2)    S1 > S2         1
```

## See also

[Summaries and plotting](../summaries.md).
