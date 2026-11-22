# information_reliability

```{eval-rst}
.. autofunction:: pyfcirt.information_reliability
```

Approximate reliability at fixed calibration using the full Fisher matrix.

## Parameters

`model` : Model
: Fitted MCMC model with a common answered form and one zero-mean Gaussian
  population. Individual nuisance effects and composite rankings are excluded.

`draw` : tuple or None
: One `(chain_label, draw_label)`. `None` uses posterior means.

`points`, `refine_points` : int
: Two powers of two for Sobol integration; the refinement must use more
  points. Defaults: 4096 and 8192.

`scrambles` : int
: At least two independent scrambled rules. Default: 2.

`random_seed` : int
: Nonnegative integration seed. Default: 42.

`chunk_size` : int
: Maximum Fisher batch size and number of population integration points
  processed together. The default is 64 for all models.

## Returns

xarray.Dataset
: Trait-indexed `empirical` and `marginal` coefficients. The empirical
  coefficient uses scores for the fitted respondents; the marginal coefficient
  integrates the fixed population. `refinement_delta` and `scramble_sd`
  describe numerical sensitivity. `marginal_by_scramble` adds `scramble`.

## Notes

The coefficient holds calibration at a point and approximates score uncertainty using local expected curvature. For covariate or multigroup models use posterior reliability.

## Examples

Use the fitted binary 2PL model from recursive scoring, whose respondents saw the same eight items.

```python
import pyfcirt as fc

result = fc.information_reliability(model)
print(result[["empirical", "marginal"]].to_dataframe().round(2).to_string())
```

```text
        empirical  marginal
trait
theta1       0.67      0.67
```

## See also

[Information and reliability](../information.md).
