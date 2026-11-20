# TwoPLRankDistribution

```{eval-rst}
.. autoclass:: pyfcirt.models.sequential.TwoPLRankDistribution
```

```{eval-rst}
.. automethod:: pyfcirt.models.sequential.TwoPLRankDistribution.dist
```

Dominance ranking distribution based on sequential selection.

## Parameters

`theta`
: Latent trait values ending in the observation's statement axis.

`loading`
: Loadings aligned with the final axes of `theta`.

`intercept`
: One subtractive intercept per statement.

`orders`
: Complete zero-based permutations of K statements, shape (K!, K). Row order determines the category order.

`membership`
: Boolean (event, complete order) matrix partitioning the full support into observed events. `None` gives one category per order.

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


Each pair of trait and signed loading values belongs to the same statement position. The output is the log probability of category index 1.

```python
import pymc as pm
from pyfcirt.models.dist import TwoPLRankDistribution
from itertools import permutations
import jax.numpy as jnp

orders = list(permutations(range(3)))
rv = TwoPLRankDistribution.dist([0.2, -0.4, 0.1], [1.0, 1.2, 0.8], [0.0, 0.2, -0.1], orders=orders)
print(round(float(pm.logp(rv, 1).eval(mode="JAX")), 6))
```

```text
-1.226257
```

### Register an observed response


Use the same arrays inside a native PyMC model. The observed variable is registered under its name.

```python
with pm.Model() as custom:
    orders = list(permutations(range(3)))
    response = TwoPLRankDistribution(
        "response", [0.2, -0.4, 0.1], [1.0, 1.2, 0.8], [0.0, 0.2, -0.1], orders=orders, observed=1
    )
print([rv.name for rv in custom.observed_RVs])
```

```text
['response']
```

### PICK from complete orders


Give one membership row per winner. Category 0 is the event that statement position 0 wins.

```python
membership = [[order[0] == winner for order in orders] for winner in range(3)]
response = TwoPLRankDistribution.dist(
    [0.2, -0.4, 0.1], [1.0, 1.2, 0.8], [0.0, 0.2, -0.1], orders=orders, membership=membership
)
print(round(float(pm.logp(response, 0).eval(mode="JAX")), 6))
```

```text
-0.873376
```

## See also

[Native distributions](../distributions.md) and [measurement declarations](../measurement.md).
