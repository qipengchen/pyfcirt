# TIRTRankDistribution

```{eval-rst}
.. autoclass:: pyfcirt.models.thurstonian.rank.TIRTRankDistribution
```

```{eval-rst}
.. automethod:: pyfcirt.models.thurstonian.rank.TIRTRankDistribution.dist
```

Marginal Gaussian response distribution for a ranking event.

## Parameters

`theta`
: Trait values ending in (statement, trait). Use one trait column per statement if each has one aligned trait.

`loading`
: Signed loadings ending in the same (statement, trait) axes.

`intercept`
: Additive statement intercepts ending in the statement axis.

`residual_variance`
: Positive residual variance for every statement, with the final statement axis.

`orders`
: Complete zero-based permutations of K statements, shape (K!, K). Row order determines the category order.

`contrasts`
: Positive adjacent differences of utility, shape (K!, K−1, K), aligned to `orders`.

`membership`
: Boolean (event, complete order) matrix partitioning the full support into observed events. `None` gives one category per order.

`n_points`
: Number of fixed Sobol nodes for the marginal integral. Default: 4096.

`seed`
: Integration-node seed. Default: 914.

`chunk_size`
: Maximum live quadrature node block. Default: 128.

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

The marginal log probability uses fixed-node quadrature. Raise `n_points` to check numerical accuracy. The high-level TIRTRank augmented sampler uses its own observed order-region graph.

## Examples

### Evaluate a response


The example enumerates all six complete orders of a triplet and computes the adjacent contrasts for each order. The output is the log probability of category index 1.

```python
import pymc as pm
from pyfcirt.models.dist import TIRTRankDistribution
from itertools import permutations
import jax.numpy as jnp

orders = list(permutations(range(3)))
eye = jnp.eye(3)
contrasts = (eye[jnp.array(orders)[:, :-1]] - eye[jnp.array(orders)[:, 1:]]).tolist()
rv = TIRTRankDistribution.dist(
    [[0.2], [-0.4], [0.1]],
    [[1.0], [1.2], [0.8]],
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
    orders=orders,
    contrasts=contrasts,
)
print(round(float(pm.logp(rv, 1).eval(mode="JAX")), 6))
```

```text
-1.133392
```

### Register an observed response


Use the same arrays inside a native PyMC model. The observed variable is registered under its name.

```python
with pm.Model() as custom:
    orders = list(permutations(range(3)))
    eye = jnp.eye(3)
    contrasts = (eye[jnp.array(orders)[:, :-1]] - eye[jnp.array(orders)[:, 1:]]).tolist()
    response = TIRTRankDistribution(
        "response",
        [[0.2], [-0.4], [0.1]],
        [[1.0], [1.2], [0.8]],
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        orders=orders,
        contrasts=contrasts,
        observed=1,
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
response = TIRTRankDistribution.dist(
    [[0.2], [-0.4], [0.1]],
    [[1.0], [1.2], [0.8]],
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
    orders=orders,
    contrasts=contrasts,
    membership=membership,
)
print(round(float(pm.logp(response, 0).eval(mode="JAX")), 6))
```

```text
-0.739375
```

## See also

[Native distributions](../distributions.md) and [measurement declarations](../measurement.md).
