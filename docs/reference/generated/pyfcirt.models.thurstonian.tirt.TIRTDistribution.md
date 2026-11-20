# TIRTDistribution

```{eval-rst}
.. autoclass:: pyfcirt.models.thurstonian.tirt.TIRTDistribution
```

```{eval-rst}
.. automethod:: pyfcirt.models.thurstonian.tirt.TIRTDistribution.dist
```

Thurstonian item response distribution for a recorded pair.

## Parameters

`theta`
: Trait values at the two recorded statement positions, final shape (..., 2).

`loading`
: Signed loadings aligned to those two positions.

`cutpoints`
: Categorical boundaries in increasing order on a final C−1 axis for C categories. For Beta, give one offset per observation.

`link`
: `probit` (default), 1.702-scaled discrete `logit`, `beta_probit` or `beta_logit`.

`statement_effect`
: Explicit effects along the two statement positions for discrete links. Omit for Beta.

`residual_variance`
: Positive variances along the statement axis. Omit for Beta.

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

Supply both `statement_effect` and `residual_variance` for a discrete link. Beta links instead require positive `precision` and exclude these inputs.

## Examples

### Evaluate a response


theta, signed loadings and discrete statement effects follow the recorded order. The output is the log probability of category index 1.

```python
import pymc as pm
from pyfcirt.models.dist import TIRTDistribution

rv = TIRTDistribution.dist(
    [0.2, -0.4],
    [1.0, 1.2],
    [0.0],
    statement_effect=[0.0, 0.0],
    residual_variance=[0.5, 0.5],
)
print(round(float(pm.logp(rv, 1).eval(mode="JAX")), 6))
```

```text
-0.285354
```

### Register an observed response


Use the same arrays inside a native PyMC model. The observed variable is registered under its name.

```python
with pm.Model() as custom:
    response = TIRTDistribution(
        "response",
        [0.2, -0.4],
        [1.0, 1.2],
        [0.0],
        statement_effect=[0.0, 0.0],
        residual_variance=[0.5, 0.5],
        observed=1,
    )
print([rv.name for rv in custom.observed_RVs])
```

```text
['response']
```

### Continuous ratings


Evaluate a Beta **log density** at a rating of 0.7 with precision 20.

```python
response = TIRTDistribution.dist([0.2, -0.4], [1.0, 1.2], 0.0, link="beta_probit", precision=20.0)
print(round(float(pm.logp(response, 0.7).eval(mode="JAX")), 6))
```

```text
1.149873
```

### Ordered categories


Two increasing cutpoints produce three categories. This evaluates the middle category.

```python
response = TIRTDistribution.dist(
    [0.2, -0.4], [1.0, 1.2], [-0.5, 0.5], statement_effect=[0.0, 0.0], residual_variance=[0.5, 0.5]
)
print(round(float(pm.logp(response, 1).eval(mode="JAX")), 6))
```

```text
-1.172551
```

## See also

[Native distributions](../distributions.md) and [measurement declarations](../measurement.md).
