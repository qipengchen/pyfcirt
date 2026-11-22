# Differential item functioning

Specify respondent characteristics that may change measurement parameters at
the same trait level. Use [population and covariate specifications](population.md)
to define group differences in the trait distribution independently.

## Add DIF terms

```python
model.add_dif_terms(
    targets, *, person_data, formula=None, groups=(), id="person_id",
    pooling="unit", prior="hierarchical", prior_options=None, overrides=None,
)
```

Call this method after preparing the model and before fitting or fixing its
parameters. Use one target string or a sequence, such as `"fc.cutpoints"`,
`"items.loading_magnitude"`, or `"fc.TIRT.cutpoints"` for a model-qualified
target. Only free, supported parameters receive effects. An ordered cutpoint
vector must be entirely free. Its responder offset shifts every boundary
together and preserves their order.

- **person_data, id** identify the respondent feature table and its ID column.
  Supply one row per respondent. Use string, boolean, or pandas categorical
  columns for groups. Set a category order to control a single factor’s
  reference. A crossed factor uses the first observed combination.
- **formula** is an optional Formulae right-hand side for continuous predictors
  and interactions. The baseline supplies the intercept.
- **groups** specifies one factor or a sequence of factors. For a joint factor
  with an offset for each observed combination, use a tuple of column names as
  an entry, for example `groups=[("site", "language")]`.
- **pooling** uses a coefficient vector per measurement unit by default.
  `"shared"` uses one vector across selected parameters with the same kind,
  prior role, and link.
- **prior** is `"hierarchical"`, `"pairwise_l1"`, or `"truncated_l1"`.
  The two L1 choices apply to the entries of `groups`. A formula supplied
  alongside those groups uses the hierarchical prior.
- **prior_options** may supply ordinary coefficient priors using `priors`,
  or a fixed or native PyMC scale using `parameter_residual_sd`. Default
  coefficients use a zero-centered normal with an automatically scaled
  slope. The local standard deviation has a HalfNormal(0.5) prior.
  The pairwise L1 default is `strength=3.0`. Truncated L1 defaults are
  `strength=1.5`, `cutoff=0.5`, and `base_sd=0.55`. These settings apply
  on the selected parameter's regression scale.
- **overrides** maps targets to `prior`, `prior_options`, or `pooling`
  overrides. A model-qualified override can refine a selected unqualified
  target. A target that selects a different prior family uses that family's
  defaults unless its override supplies options.

The method returns `None`. A subsequent declaration for the same target and
formula replaces that effect. Other terms and the ordinary baseline are
retained. A failing declaration preserves the original model configuration.
For scoring or simulation with new respondents, supply their covariate
values; a precomputed generation schedule retains its originating encoding.
When restoring a saved fit, redeclare the original DIF terms before
[attaching the saved samples](results.md#save-and-restore-samples).

A common trait scale across groups needs measurement anchors or enough
parameters that are approximately invariant. A broad shift in thresholds
in the same direction can be difficult to distinguish from a shift in the
population trait mean.

The DIF defaults were set using prior predictions for two and three groups
and separate simulated TIRT fits. For two groups, the default hierarchical,
ordinary L1, and truncated L1 priors put approximately 14%, 14%, and 32% of
pair threshold differences above 0.65 in absolute value. The three selected
defaults each detected 18 of 19 planted threshold contrasts, with zero
false positives, in five independent tuning datasets. That comparison
included balanced, unbalanced, and three-group designs. Use the
measurement parameter's scale when choosing prior overrides.

## Summarize DIF

```python
model.dif_summary(contrasts=None, *, at=None, prob=0.95, significant_only=False)
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

```python
model.add_dif_terms("fc.cutpoints", person_data=respondents, groups="site")
model.fit(random_seed=42)
model.dif_summary(contrasts={"site": ("A", "B")})
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
The truncated law checks numerical integration and feasibility of forward
rejection sampling during construction. The integrated proposal acceptance
must be at least 0.1%. Specify at least two groups.
