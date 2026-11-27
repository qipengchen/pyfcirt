# Differential item functioning

**Experimental.** The DIF API and defaults may change.

Specify respondent characteristics that may change measurement parameters at
the same trait level. Use [population and covariate specifications](population.md)
to define group differences in the trait distribution independently. For DIF
in forced-choice TIRT models, see Lee et al. (2021).

## Add DIF terms

```text
model.add_dif_terms(
    targets, *, formula=None, person_data=None, reference_levels=None,
    prior="laplace", prior_options=None, pooling="unit", id="person_id", groups=(),
)
```

Call this method after preparing the model and before fitting or fixing its
parameters. Use one target string or a sequence, such as `"fc.cutpoints"`,
`"items.loading_magnitude"`, or `"fc.TIRT.cutpoints"` for a model-qualified
target. Pass a mapping to configure individual targets. Its values accept `prior`,
`prior_options`, `pooling`, and `anchors`. Target settings take precedence over
call-wide defaults. Only free, supported parameters receive effects. An ordered cutpoint
vector must be entirely free. Its responder offset shifts every boundary
together and preserves their order.
Use [available_covariate_targets(kind="dif")](generated/pyfcirt.Model.available_covariate_targets.md)
to see the eligible targets for the prepared design.

- **person_data, id** identify the respondent feature table and its ID column.
  Omit the table to use the person data registered before preparation.
  Supply one row per respondent. Use string, boolean, or pandas categorical
  columns for groups.
- **formula** is a Formulae right-hand side, such as `site`, `site * language`,
  or `site * age`. The baseline supplies the intercept.
- **reference_levels** maps categorical predictor names to their reference
  levels. An explicit level takes precedence over the population reference
  for the same categorical column, then over the first encoded category.
  Use the same label type as in the data. For joint categories, supplied
  levels select the first observed combination that matches. A numeric
  population group column remains numeric in a DIF formula unless coded as
  categorical. All three shrinkage priors leave the encoded reference at
  the measurement baseline. Pairwise penalties treat group differences
  symmetrically.
- **groups** specifies one factor or a sequence of factors. For a joint factor
  with an offset for each observed combination, use a tuple of column names as
  an entry, for example `groups=[("site", "language")]`.
- **pooling** uses a coefficient vector per measurement unit by default.
  `"shared"` uses one vector across selected parameters with the same kind,
  prior role, and link.
- **prior** defaults to zero-centered `"laplace"` for the formula coefficients.
  A Laplace prior on the coefficients gives L1 shrinkage (Park & Casella,
  2008). Supply a scalar PyMC distribution to set the coefficient prior directly.
  `"pairwise_l1"` and `"truncated_l1"` apply to a single categorical formula
  term or an explicit group term, including when both are used together.
  Formulas with numeric predictors use Laplace when a pairwise prior is
  selected. Use the explicit `hierarchical` setting to reproduce earlier
  models. Experimental `"spike_and_slab"` gives each free coefficient a point
  mass at zero and a zero-centered Normal slab. Set `inclusion_prob` (default
  0.2) and `slab_sd` (default 1.0 on the regression scale) in `prior_options`.
  Fit with `nuts_sampler="pymc"` to use native compound sampling for the
  binary inclusion indicators and continuous parameters.
- **prior_options** sets Laplace `b` or the chosen pairwise parameters.
  The automatic Laplace scales are 1/6 for loading effects and 1/3 for other
  effects, divided by the fitted continuous predictor scale for numeric terms.
  Smaller `b` gives stronger Laplace shrinkage. Larger `strength` gives
  stronger pairwise shrinkage.
  Ordinary pairwise L1 defaults to `strength=3.0`. For truncated L1 on
  loading magnitudes, the default strength is `3 * sqrt(N / 800) * (2 / S)`,
  where `N` is the number of respondents in the fitted design and `S` is
  the number of represented groups for that DIF term. For an explicit
  crossed term, `S` counts observed group combinations. The strength is
  3.0 for 800 respondents in two groups. Other targets use `strength=1.5`
  under truncated L1, which uses `cutoff=0.5` and `base_sd=0.55` for all
  targets by default. An explicit `strength` overrides the default.
  These settings apply on the selected parameter's regression scale.
- **anchors** in a target configuration identify statement IDs, item IDs,
  or comparison keys `(block_id, statement_1, statement_2)`. The selected
  parameters share their estimated baseline across all levels of the
  declared DIF term. Reverse comparison keys select the same pair.

The method returns `None`. A subsequent declaration for the same target and
formula replaces that effect. Other terms and the ordinary baseline are
retained. A failing declaration preserves the original model configuration.
For scoring or simulation with new respondents, supply their covariate
values; a precomputed generation schedule retains its originating encoding.
When restoring a saved fit, redeclare the original DIF terms before
[attaching the saved samples](results.md#save-and-restore-samples).

A common trait scale across groups needs measurement anchors or enough
parameters that are approximately invariant. With free focal-group SDs and
unanchored loadings, multiplying a focal trait by a positive constant and
dividing its loadings by the same constant leaves response probabilities
unchanged. The mean and SD of that trait scale by the same factor. The
distinction relies on the prior assumption of approximate invariance. Likewise, a broad threshold shift may be
attributed to the population trait mean. Chen et al. (2023) discuss
identification under sparse DIF when anchor items are unknown.

## Summarize DIF

```text
model.dif_summary(
    contrasts=None, *, at=None, prob=0.95, significant_only=False,
    bayes_factor=False, prior_draws=4000, random_seed=None, assume_same_prior=False,
)
```

Return a pandas table of posterior differences on the measurement parameter's
original scale. Rows name the parameter, its measurement units, the predictor,
and the levels or values compared. `reference_mean` and `compared_mean` are
the fitted means at those profiles. `mean`, `lower`, and `upper` summarize the
to-minus-from difference. `detected` indicates whether the equal-tailed
credible interval excludes zero.

By default the table reports group pairs and one-unit numeric contrasts from
each target's fitted levels and centers. Use `contrasts={"age": (20, 30)}` or
`contrasts={"site": ("A", "B")}` to select comparison endpoints. A crossed
factor uses a key such as `"site:language"` and two tuples in the order given
when declaring the factor. Predictors outside a contrast stay at their trained
center or reference level unless overridden with `at`. Crossed profiles
must be combinations observed in the training data. Set
`significant_only=True` to keep detected rows. The interval rule is for
individual comparisons.

### Per-parameter Bayes factors

Set `bayes_factor=True` to add Savage–Dickey Bayes factors for one two-group
TIRT contrast and independent, fixed-scale coefficient priors. Supply one
categorical comparison through `contrasts`. Supported priors are zero-centered
Normal or Laplace, Pairwise L1, and truncated Pairwise L1. Each reported target
must have a single free coefficient for the requested comparison. Anchors and
fixed parameters have no free contrast to test.

`BF10` favors a free group effect over a zero effect. `BF01` favors the zero
effect over the free effect. `bf_scale` identifies the scale used for evidence.
Loading evidence uses the difference of log loadings. Cutpoint evidence uses
the cutpoint difference. The interval columns still use the original
measurement scales, and `significant_only` still filters by the interval rule.

`prior_draws=4000` controls draws from the fitted coefficient priors and must
be at least 40.
`random_seed` makes prior sampling reproducible. Both the prior and posterior
draws use the same fitted encoder and contrast. To reuse an older or externally
attached posterior without recorded prior identity, set `assume_same_prior=True`
only after verifying that its original coefficient priors and bindings match.
The check uses recorded prior identity when available.

`bf_status` reports `ok`, `few_draws_near_zero`, or `invalid_density`. For
draws that bracket zero, fewer than 20 on either side in either distribution
marks limited information and retains any finite estimate. If zero lies
outside the density grid, the BF and status are empty. The effect estimate
and interval remain available.

For a fitted `"spike_and_slab"` term, the table also reports
`inclusion_probability`, the posterior fraction of draws in the slab.
`BF10` is posterior inclusion odds divided by prior inclusion odds.
Point-zero evidence uses the effect draws directly. `indicator_state_unobserved`
marks an unvisited inclusion state and omits the BF. `few_indicator_draws`
marks fewer than 20 draws in either state.

```python
model.dif_summary(
    contrasts={"site": ("A", "B")},
    bayes_factor=True,
    random_seed=42,
)
```

## Examples

These examples use the TIRT design and binary pair responses from the
[TIRT DIF tutorial](../tutorials/tirt_dif.ipynb). The person table has one
row per respondent, including `person_id` and a categorical `site` column
with levels A and B. The interaction example also uses a categorical
`language` column with levels en and zh observed at both sites. Use a freshly
prepared model for each alternative DIF declaration.

### Set the population

A is the reference population with zero trait means and unit residual SDs.
The population default estimates a covariance matrix and trait means for
each focal group. To keep unit residual SDs and share the correlation matrix,
set `theta_residual_scale="unit"` and `theta_correlation="shared"`.

```python
model = fc.Model()
model.add_fc(design, responses, format="block_long", model="TIRT", response_format=None)
model.set_population(group="site", reference_group="A", person_data=people)
model.prepare()
```

### Check loading and cutpoint DIF

Both targets use the default Laplace prior and A as the reference level.

```python
model.add_dif_terms(["fc.loading_magnitude", "fc.cutpoints"], formula="site")
model.fit(random_seed=42)
```

```python
model.dif_summary(contrasts={"site": ("A", "B")})
```

### Anchor a comparison

The two statement loadings and the cutpoint of the same pair are estimated
as common across sites.

```python
model.add_dif_terms(
    {
        "fc.loading_magnitude": {"anchors": ["B01_1", "B01_2"]},
        "fc.cutpoints": {"anchors": [("B01", "B01_1", "B01_2")]},
    },
    formula="site",
)
```

### Give each target a different prior

The loading effects use Laplace with scale 0.15 on the log scale. All cutpoint
group pairs use the ordinary L1 prior with strength 3. The values illustrate
how to set a prior on each target.

```python
model.add_dif_terms(
    {
        "fc.loading_magnitude": {"prior_options": {"b": 0.15}},
        "fc.cutpoints": {"prior": "pairwise_l1", "prior_options": {"strength": 3.0}},
    },
    formula="site",
)
```

### Compare sites within each language

For a teaching illustration, assign two synthetic language labels within each
site. With real data, supply the observed language column instead.

```python
people = people.assign(
    language=people.groupby("site", observed=True).cumcount().mod(2).map({0: "en", 1: "zh"})
)
```

```python
model.add_dif_terms(
    ["fc.loading_magnitude", "fc.cutpoints"],
    formula="site * language",
    person_data=people,
    reference_levels={"site": "A", "language": "en"},
)
model.fit(random_seed=42)
```

```python
model.dif_summary(contrasts={"site": ("A", "B")}, at={"language": "en"})
```

```python
model.dif_summary(contrasts={"site": ("A", "B")}, at={"language": "zh"})
```

## PairwiseL1

```python
from pyfcirt.models.dist import PairwiseL1

prior = PairwiseL1.dist(groups=3, strength=1.0, cutoff=None, base_sd=1.0)
```

The unnamed PyMC variable has a `groups - 1` contrast event axis. Calling
`PairwiseL1("name", ...)` gives the same distribution a name inside a native
PyMC model. The prior is proportional to the exponential of the negative sum
of pairwise absolute differences between zero-sum group effects.
`cutoff=None` gives ordinary L1. A finite positive cutoff caps each
difference's penalty and adds a normal base with scale `base_sd`.
Lyu et al. (2025) propose the truncated all-pairs penalty for multi-group DIF.
The truncated law checks numerical integration and feasibility of forward
rejection sampling during construction. The integrated proposal acceptance
must be at least 0.1%. Specify at least two groups.

## References

Chen, Y., Li, C., Ouyang, J., & Xu, G. (2023). DIF statistical inference without knowing anchoring items. *Psychometrika, 88*(4), 1097–1122. [https://doi.org/10.1007/s11336-023-09930-9](https://doi.org/10.1007/s11336-023-09930-9)

Lee, P., Joo, S.-H., & Stark, S. (2021). Detecting DIF in multidimensional forced choice measures using the Thurstonian item response theory model. *Organizational Research Methods, 24*(4), 739–771. [https://doi.org/10.1177/1094428120959822](https://doi.org/10.1177/1094428120959822)

Lyu, W., Wang, C., & Xu, G. (2025). Detecting differential item functioning across multiple groups using group pairwise penalty. *Psychometrika, 90*(5), 1594–1621. [https://doi.org/10.1017/psy.2025.10034](https://doi.org/10.1017/psy.2025.10034)

Park, T., & Casella, G. (2008). The Bayesian lasso. *Journal of the American Statistical Association, 103*(482), 681–686. [https://doi.org/10.1198/016214508000000337](https://doi.org/10.1198/016214508000000337)
