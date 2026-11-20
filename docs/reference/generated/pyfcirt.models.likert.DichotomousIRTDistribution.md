# DichotomousIRTDistribution

```{eval-rst}
.. autoclass:: pyfcirt.models.likert.DichotomousIRTDistribution
```

```{eval-rst}
.. automethod:: pyfcirt.models.likert.DichotomousIRTDistribution.dist
```

Binary logistic item response distribution with optional asymptotes.

## Parameters

`theta`
: Ability values along the final trait axis.

`loading`
: Loadings aligned with the final axes of `theta`.

`intercept`
: Subtractive intercept for each item/observation, without a trait axis.

`lower`
: Lower asymptote, default 0.0.

`upper`
: Upper asymptote, default 1.0. Both must satisfy 0 ≤ lower < upper ≤ 1.

`size`
: Native PyMC batch size. Choose either `size` or `shape`.

`shape`
: Output shape, alternative to `size`. Parameter axes must be known.


For the named form, also supply `name` and optional PyMC arguments including
`observed` and `dims`. Parameter lengths must be known; a `dims` label
does not infer the shape of a symbolic parameter.

## Returns

TensorVariable
: An unnamed PyMC response RV from `.dist()`. The named constructor
  registers the response in the model. When evaluating or supplying responses,
  category indices start at 0.

## Notes

For the default 2PL asymptotes omit `lower` and `upper`; a common 1PL loading is supplied by the caller.

## Examples

### Evaluate a response


The item measures one trait and has lower and upper asymptotes of 0.1 and 0.9. The output is the log probability of category index 1.

```python
import pymc as pm
from pyfcirt.models.dist import DichotomousIRTDistribution

rv = DichotomousIRTDistribution.dist([0.2], [1.2], 0.1, lower=0.1, upper=0.9)
print(round(float(pm.logp(rv, 1).eval(mode="JAX")), 6))
```

```text
-0.638745
```

### Register an observed response


Use the same arrays inside a native PyMC model. The observed variable is registered under its name.

```python
with pm.Model() as custom:
    response = DichotomousIRTDistribution(
        "response", [0.2], [1.2], 0.1, lower=0.1, upper=0.9, observed=1
    )
print([rv.name for rv in custom.observed_RVs])
```

```text
['response']
```

## See also

[Native distributions](../distributions.md) and [measurement declarations](../measurement.md).
