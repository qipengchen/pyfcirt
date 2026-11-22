# TwoPLRank

```{eval-rst}
.. autoclass:: pyfcirt.TwoPLRank
```

Sequential choice with monotone statement endorsement.

## Parameters

`priors` : mapping or None
: Override local prior roles with unnamed PyMC distributions such as
  `pm.Normal.dist(mu=0, sigma=1)`. `None` uses the defaults below.

`reference_statements` : tuple
: Statement IDs to use as zero-intercept references. Default: `()` for
  preparation's choice.

## Returns

TwoPLRank
: A measurement declaration for `FCInput(..., model=declaration)`. Pass the configured
  declaration as the `model` value when loading a bundled dataset.

## Notes

Choose RANK, PICK or MOLE in the input. Full ranks multiply the probabilities
 of selecting each statement from those remaining. For a partial event the
 law sums over compatible complete orders.

Default priors for active roles:

| Prior role | Default law |
| --- | --- |
| `loading_magnitude` | LogNormal(mu=0, sigma=0.5) |
| `ranking_intercept` | Normal(mu=0, sigma=2) |

## Examples

```python
import pyfcirt as fc

declaration = fc.TwoPLRank()
example = fc.load_example("simulated_two_pl_rank", model=declaration, n_persons=3)
prepared = fc.PreparedInputs.from_inputs(example.data)
print(fc.data_coverage(prepared).to_string())
```

```text
                       n_blocks  size  n_persons  n_answered  n_missing
model_response_format
TwoPLRank / rank             24     3          3          72          0
```

## See also

[Equations and identification](../../models/two_pl_rank.md) and
[worked analysis](../../tutorials/two_pl_rank_workflow.ipynb).
