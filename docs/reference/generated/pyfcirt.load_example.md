# load_example

```{eval-rst}
.. autofunction:: pyfcirt.load_example
```

Load a bundled dataset for a chosen measurement model.

## Parameters


`name` : str
: Name in the [dataset catalogue](../../datasets.md).

`model` : str, measurement specification or mapping
: Current measurement model. For mixed data, supply
  `{"fc": fc_model, "items": item_model}` with one model per channel.

`n_persons` : int or None
: Number of people to retain. `None` uses the resource's default cohort.
  The full design and category scales are retained for any subset.

`random_seed` : int or None
: With explicit `n_persons`, sample people without replacement. The seed
  is in `[0, 2**32 - 1]`. `None` takes the first source respondents.

`include_truth` : bool
: Include original generating values for simulated resources. Default: `False`.


## Returns

ExampleData
: Fresh inputs, matched optional `person_data` and `response_times` tables,
  provenance `metadata` and optional `truth`. The selected cohort filename
  is recorded in `metadata["selection"]["resource"]` as `n<N>.json.gz` (for
  example, `n100.json.gz`). Pass `bundle.data` to
  `Model.prepare()`. Companion tables retain source row order, so match by ID.

## Notes

Generating values use the source model's parameter names and scale. Empirical
resources return `truth=None`. The `hexaco_fc3` resource also provides
one independent desirability score per statement in
`bundle.data.statement_features`, matched by `statement_id`. The
`mgppm_groups` resource provides respondent group labels in `bundle.person_data`.

## Examples

### Observed comparisons


```python
import pyfcirt as fc

bundle = fc.load_example("tirt_study", model="TIRT", n_persons=3)
print(bundle.data.responses.frame.head(3).to_string(index=False))
```

```text
person_id block_id statement_1 statement_2 response
       P1       B1          S1          S2        0
       P1       B1          S1          S3        1
       P1       B1          S2          S3        1
```

### Simulated responses with generating parameters


```python
bundle = fc.load_example(
    "simulated_mirt",
    model="DichotomousIRT",
    n_persons=50,
    include_truth=True,
)
print({axis: bundle.truth.sizes[axis] for axis in ("person", "item")})
```

```text
{'person': 50, 'item': 8}
```

## See also

[ExampleData](pyfcirt.ExampleData.md) and [all dataset names](../../datasets.md).
