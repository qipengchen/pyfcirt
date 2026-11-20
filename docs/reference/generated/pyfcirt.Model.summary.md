# Model.summary

```{eval-rst}
.. automethod:: pyfcirt.Model.summary
```

Describe a prepared model and its priors and fixed values.

## Parameters

`details` : bool
: Include expanded counts and model details. Default: `False`.

## Returns

str
: Declared models, observation counts, trait population, priors and fixed
  references. Use `coef()` for posterior estimates.

## Examples

Check that one loading is shared across eight binary items.

```python
import pyfcirt as fc

bundle = fc.load_example(
    "simulated_mirt",
    model=fc.DichotomousIRT(kind="1PL"),
    n_persons=3,
)
model = fc.Model(bundle.data)
print(model.summary())
```

```text
Model: DichotomousIRT
3 people, 1 traits, 8 items
Observations: 24 item responses
Measurements:
  8 items, link=logit
Population:
  theta ~ MvNormal(mu=0, cov=1)
Priors:
    loading [1] ~ LogNormal(mu=0.25, sigma=0.5)
    item_intercept [8] ~ Normal(mu=0, sigma=1)
Constraints:
    lower_asymptote: fixed=0 [8 fixed]
    upper_asymptote: fixed=1 [8 fixed]
```

## See also

[Summaries and plotting](../summaries.md).
