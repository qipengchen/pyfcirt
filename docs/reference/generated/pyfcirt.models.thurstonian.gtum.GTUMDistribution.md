# GTUMDistribution

```{eval-rst}
.. autoclass:: pyfcirt.models.thurstonian.gtum.GTUMDistribution
```

```{eval-rst}
.. automethod:: pyfcirt.models.thurstonian.gtum.GTUMDistribution.dist
```

Unfolding response distribution for a recorded pair.

## Parameters

`theta`
: Trait values at the two recorded statement positions, final shape (..., 2).

`loading`
: Positive loading magnitudes for the two statements.

`location`
: Signed ideal-point locations aligned with `theta` and `loading`.

`cutpoints`
: Categorical boundaries in increasing order on a final C−1 axis for C categories. For Beta, give one offset per observation.

`link`
: `logit` (default) or `beta_logit`; neither uses a residual scale or link multiplier.

`person_block_effect`
: Scalar or observation-aligned effect for the discrete model. Default: 0.

`precision`
: Positive Beta concentration. Supply it for a Beta link.

`direction`
: +1 (default) makes higher values favor the first recorded statement; −1 favors the second.

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
  category indices start at 0; use the actual rating when the link is Beta.

## Notes

The block effect enters the discrete utility difference before applying direction. Use zero for Beta.

## Examples

### Evaluate a response


Positive loading magnitudes and actual signed locations follow the recorded order. The output is the log probability of category index 1.

```python
import pymc as pm
from pyfcirt.models.dist import GTUMDistribution

rv = GTUMDistribution.dist([0.2, -0.4], [1.0, 1.2], [1.0, -1.0], [0.0])
print(round(float(pm.logp(rv, 1).eval(mode="JAX")), 6))
```

```text
-0.733947
```

### Register an observed response


Use the same arrays inside a native PyMC model. The observed variable is registered under its name.

```python
with pm.Model() as custom:
    response = GTUMDistribution("response", [0.2, -0.4], [1.0, 1.2], [1.0, -1.0], [0.0], observed=1)
print([rv.name for rv in custom.observed_RVs])
```

```text
['response']
```

### Continuous ratings


Evaluate a Beta **log density** at a rating of 0.7 with precision 20.

```python
response = GTUMDistribution.dist(
    [0.2, -0.4], [1.0, 1.2], [1.0, -1.0], 0.0, link="beta_logit", precision=20.0
)
print(round(float(pm.logp(response, 0.7).eval(mode="JAX")), 6))
```

```text
-0.665153
```

### Ordered categories


Two increasing cutpoints produce three categories. This evaluates the middle category.

```python
response = GTUMDistribution.dist([0.2, -0.4], [1.0, 1.2], [1.0, -1.0], [-0.5, 0.5])
print(round(float(pm.logp(response, 1).eval(mode="JAX")), 6))
```

```text
-1.408333
```

## See also

[Native distributions](../distributions.md) and [measurement declarations](../measurement.md).
