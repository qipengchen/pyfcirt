# response_summary

```{eval-rst}
.. autofunction:: pyfcirt.response_summary
```

Count observed responses and compute proportions.

## Parameters

`data` : pandas.DataFrame
: Long answer table returned by `response_data()`.

`groupby` : str, sequence of str or None
: Optional extra group columns. The default groups by the observed unit and
  pair, where applicable.

## Returns

pandas.DataFrame
: One row per observed category in each unit, pair and selected group.
  `count` is its frequency, `n_valid` the denominator and `proportion`
  their ratio.

## Examples

Count the two outcomes for each statement pair in B1.

```python
import pyfcirt as fc

bundle = fc.load_example("tirt_study", model="TIRT", n_persons=6)
model = fc.Model(bundle.data)

answers = model.response_data(blocks=["B1"])
counts = fc.response_summary(answers)
columns = ["pair_label", "category", "count", "n_valid", "proportion"]
print(counts[columns].round(2).to_string(index=False))
```

```text
pair_label  category  count  n_valid  proportion
   S1 > S2         0      2        6        0.33
   S1 > S2         1      4        6        0.67
   S1 > S3         0      3        6        0.50
   S1 > S3         1      3        6        0.50
   S2 > S3         0      3        6        0.50
   S2 > S3         1      3        6        0.50
```

## See also

[Summaries and plotting](../summaries.md).
