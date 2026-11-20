# GGUMDistribution

```{eval-rst}
.. autoclass:: pyfcirt.models.likert.GGUMDistribution
```

```{eval-rst}
.. automethod:: pyfcirt.models.likert.GGUMDistribution.dist
```

Generalized graded unfolding response distribution for one item.

## Parameters

`theta`
: Ability values along the final trait axis.

`loading`
: Positive magnitudes on the final trait axis.

`location`
: Signed ideal-point locations aligned with `theta` and `loading`.

`threshold`
: Full category step vector on the final axis. First step is zero; remaining steps can be positive or negative.

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

The caller supplies identification and any prior distributions.

## Examples

### Evaluate a response


The example specifies three response categories and an ideal point of 1.0. The output is the log probability of category index 1.

```python
import pymc as pm
from pyfcirt.models.dist import GGUMDistribution

rv = GGUMDistribution.dist([0.2], [1.2], [1.0], [0.0, 0.4, -0.2])
print(round(float(pm.logp(rv, 1).eval(mode="JAX")), 6))
```

```text
-1.734589
```

### Register an observed response


Use the same arrays inside a native PyMC model. The observed variable is registered under its name.

```python
with pm.Model() as custom:
    response = GGUMDistribution("response", [0.2], [1.2], [1.0], [0.0, 0.4, -0.2], observed=1)
print([rv.name for rv in custom.observed_RVs])
```

```text
['response']
```

## See also

[Native distributions](../distributions.md) and [measurement declarations](../measurement.md).
