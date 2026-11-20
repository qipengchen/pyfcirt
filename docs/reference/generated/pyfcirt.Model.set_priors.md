# Model.set_priors

```{eval-rst}
.. automethod:: pyfcirt.Model.set_priors
```

Override native priors on selected parameters before fitting.

## Parameters

`priors` : mapping of str to PyMC random variable
: Map a prepared role selector to an unnamed native distribution such as
  `pm.Normal.dist(mu=0, sigma=1)`.

| Selector | Parameters |
| --- | --- |
| `fc.loading_magnitude`, `items.loading_magnitude` | Free statement or item loadings |
| `fc.TIRT.cutpoints` | TIRT cutpoints in a mixed forced-choice design |
| `fc.residual_variance` | Free Gaussian statement variance in TIRT or TIRTRank |
| `population.trait_correlation` | Residual trait correlation |
| `population.population_mean`, `population.population_sd` | Free population means or scales |

Use short keys such as `cutpoints` inside a measurement declaration's own
`priors=` mapping. The [model pages](../measurement.md) list local roles.

## Returns

None
: Updates an unfitted model. A failed update preserves the previous specification.

## Notes

A prior applies to free entries and can replace the selected role's regression.
Fixed references remain. For a standard deviation, variance or precision,
choose a distribution with positive support.

### Priors and identification

Unfolding location sides +1, −1 and 0 use `location_positive`,
`location_negative` and `location_free`, respectively. Zero selects an
unrestricted location. Custom signed priors retain their support and may
trigger `IdentificationWarning`.

### TIRT loading anchors

For an all-cross-trait two-trait TIRT design, fix both loading magnitudes of
a reference pair. Give `TIRT(loading_anchors={"left": 1.0, "right": 1.0})`
the actual statement IDs, or use the design's `fix_loading` column.
TIRTRank also accepts `loading_anchors`.

### Shared statements in linear ranking models

Overlapping MGPPM or TwoPLRank blocks share loadings and intercepts when
they use the same statement IDs. Preparation resolves intercept references
for the connected statements.

## Examples

Update one prior and check the resolved scale in the model summary.

```python
import pymc as pm
import pyfcirt as fc

bundle = fc.load_example("tirt_study", model="TIRT", n_persons=3)
model = fc.Model(bundle.data)
model.set_priors({"fc.loading_magnitude": pm.LogNormal.dist(mu=0, sigma=0.35)})
print(next(line.strip() for line in model.summary().splitlines() if "0.35" in line))
```

```text
loading [72] ~ LogNormal(mu=0, sigma=0.35)
```

## See also

[Prior comparison](../../tutorials/prior_comparison.ipynb) and
[regression priors](pyfcirt.Model.add_regressions.md).
