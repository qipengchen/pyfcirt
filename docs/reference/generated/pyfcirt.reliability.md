# reliability

```{eval-rst}
.. autofunction:: pyfcirt.reliability
```

Compute empirical reliability from draws of the fitted person scores.

## Parameters

`model` : Model
: Fitted model with final `theta(chain, draw, person, trait)`. MCMC
  or explicitly sourced approximate draws can provide uncertainty.

`person_ids` : sequence or None
: IDs of respondents to include. Use at least two. `None` includes all.

`method` : {"ratio", "subtraction"}
: For score sample variance V and mean posterior variance U, use
  `V/(V+U)` (default) or `1-U/V`.

## Returns

xarray.Dataset
: `reliability`, `score_variance` and `mean_posterior_variance` by
  trait when that dimension is present; `n_person` is a scalar count.

## Notes

The coefficient describes the selected respondents. A point estimate of theta alone does not supply posterior variance.

## Examples

Use a fitted `Model`. The output below comes from the [HEXACO TIRT example](../../tutorials/tirt_workflow.ipynb).
Summarize posterior score reliability for each trait.

```python
result = model.reliability()
print(result["reliability"].to_series().round(2).to_string())
```

```text
trait
honestyhumility      0.74
openness             0.66
agreeableness        0.60
emotionality         0.68
conscientiousness    0.70
extraversion         0.77
```

## See also

[Information and reliability](../information.md).
