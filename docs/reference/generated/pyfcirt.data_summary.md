# data_summary

```{eval-rst}
.. autofunction:: pyfcirt.data_summary
```

Count administrations, recorded answers and likelihood events before fitting.

## Parameters


`data` : Model or PreparedInputs
: Prepared design and responses. A posterior is optional.

`by` : {"model", "block", "item"}
: Group units by model/response format (default), or give one row per
  block or item. Block and item views select the corresponding channel.

`details` : bool
: Include response support, link, refusal and nonadministration counts, and
  likelihood-event count. Default: `False`.


## Returns

pandas.DataFrame
: Index is `model_response_format`, `block` or `item`. Counts include
  people and answered or missing administrations. People appearing in several
  units of a model group are counted once. With `details=True`:

  - `n_response_categories` counts possible answers to the original task.
  - `n_categories` counts outcomes per encoded likelihood factor. Multiple
    support sizes appear as a tuple. Both category counts are missing for Beta responses.
  - `n_obs` counts likelihood events, which may outnumber answered blocks.

## Examples

```python
import pyfcirt as fc

example = fc.load_example("tirt_study", model="TIRT", n_persons=3)
model = fc.Model().prepare(example.data)
summary = fc.data_summary(model, by="block", details=True)
columns = ["n_answered", "n_response_categories", "n_categories", "n_obs"]
print(summary.loc[["B1"], columns].to_string())
```

```text
       n_answered  n_response_categories  n_categories  n_obs
block
B1              3                      2             2      9
```

## See also

[response_data()](pyfcirt.response_data.md) to inspect individual rows.
