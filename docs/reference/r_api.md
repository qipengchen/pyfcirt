# R API reference

Use `library(pyfcirt)` to load the R interface. Create a model with
`model()` and call its methods with `m$`. Integers such as `42L` supply
Python integer arguments. For installation and a complete fit, see the
[R Quick start](../guide/getting_started.md#quick-start-in-r).

## Prepare and fit

```r
model()
load_example(name, model, n_persons = NULL, random_seed = NULL, include_truth = FALSE)
m$add_fc(design, responses, format, model, response_format = "rank",
         person_id_column = NULL, ...)
m$add_items(design, responses, format, model, kind = NULL,
            categories = c(0L, 1L), person_id_column = NULL, ...)
m$prepare(..., population = NULL)
m$set_priors(priors)
m$add_regressions(formulas, ...)
m$fit(..., var_names = NULL)
```

- `model()` creates an empty analysis. `load_example()` returns a list with
  `design` and `responses` data frames and the table `format`. It also supplies
  the FC `response_format` and `scale` or the item `categories`. Mixed examples
  put these fields under `fc` and `items`. `include_truth = TRUE` requests
  generating parameters when available.
- Pass a design data frame or string, a response data frame and a model name to
  `add_fc()` or `add_items()`. For example, `format = "comparison_long"` uses
  the columns `person_id`, `block_id`, `statement_1`, `statement_2` and
  `response`. Set `response_format` to `"rank"`, `"pick"`, `"mole"` or
  `NULL` for direct comparisons. Use `kind` for `DichotomousIRT` variants.
- `prepare()` validates and aligns the registered tables. Optional arguments
  include `person_data = NULL` and
  `theta_parameterization = "centered"`. For multiple groups, pass
  `population = list(group = "school", reference_group = "A")` and a
  `person_data` table with `person_id` and `school`. Include
  `regression_by_group = TRUE` in the population list for group-specific slopes.
  `set_priors()` accepts a named list
  of native PyMC distributions. `add_regressions()` accepts a formula string
  and optional predictor tables such as `person_data` or `statement_data`.
- `fit()` attaches posterior samples to the same model. Its forwarded defaults
  include `draws = 1000L`, `tune = 1000L`, `chains = NULL`,
  `target_accept = 0.8`, `nuts_sampler = "blackjax"` and `random_seed = NULL`.
  Use `retain_calibration = TRUE` during fitting if you plan to use posterior
  calibration for scoring. The configuration and fitting methods return the
  same model invisibly.

## Inspect the data and posterior

```r
m$data_coverage(by = "model", details = FALSE)
m$response_data(blocks = NULL, items = NULL, representation = "auto", groups = NULL)
response_counts(answers, groupby = NULL)
summary(m, details = FALSE)
coef(m, blocks = NULL, items = NULL, component = "measurement",
     ci_prob = 0.95, ci_kind = "hdi", var_names = NULL, coords = list(),
     kind = "all", digits = 3L, ...)
m$trait_summary(by_group = FALSE, ci_prob = 0.95, ci_kind = "hdi", digits = 3L, ...)
as_draws(m, variables = NULL, coords = list(), chain = NULL, draw = NULL)
```

- `data_coverage()` reports respondents, available responses and missingness.
  Use `by = "block"` or `by = "item"` for a breakdown by measurement unit.
  `response_data()` returns observed records. For additional grouping, pass
  a data frame with a `person` column and one column per group variable.
  `response_counts()` takes those records and returns counts, valid totals
  and proportions. Use `groupby` to add columns to the grouping.
- `summary()` returns a printable model description. `coef()` returns a
  data frame with parameter labels and posterior statistics. In mixed tests,
  `channel` distinguishes FC blocks from individual items in `unit`.
  Group-specific estimates include `population_group`. Set
  `component = "regression"` for regression coefficients or
  `ci_kind = "eti"` for equal-tailed intervals. `digits` controls printing.
  `trait_summary()` returns a named list of population trait summaries.
- `as_draws()` returns a `posterior::draws_array` with dimensions
  iteration × chain × variable. `coords` accepts dimension labels and
  `block`, `statement`, `pair`, `item`, or `trait` labels. Use the pair
  text shown by `coef()`, such as `"S1 > S2"`. Multiple labels within one
  field are alternatives, while different fields are intersected. Shared
  parameters appear once. Variables without matching entries are omitted.
  A selection with no matching variable raises an error.
  `chain` and `draw` select original sample labels. Use the `coordinates`
  and `sample_coordinates` attributes for the original labels. Variable
  names use the original dimension labels and stay stable across selections.
  Repeated labels in a selector raise an error. Integers outside R's exact
  numeric range are stored as `integer:` labels in coordinate metadata. Reuse
  those labels in selectors for the original integer coordinate.

For the 90-person calibration in the R Quick start:

```r
m$data_coverage()$n_persons
# [1] 90

answers_B1 <- m$response_data(blocks = "B1")
counts_B1 <- response_counts(answers_B1)
sum(counts_B1$count) == nrow(answers_B1)
# [1] TRUE

loading <- as_draws(m, variables = "loading", chain = 0L, draw = 0L)
dim(loading)[1:2]
# [1] 1 1

as_draws(m, variables = "loading", coords = list(block = "B1", statement = "S1"))
as_draws(m, variables = "cutpoint", coords = list(block = "B1", pair = "S1 > S2"))
```

## Predict, check and compare

```r
m$prior_predictive(...)
m$posterior_predictive(...)
predictive_view(predictions, block = NULL, item = NULL)
m$chi_ppp(blocks = NULL, items = NULL, ...)
plot_ppp(check, coords = list(), bins = 30L, quantile_range = NULL, max_panels = 12L)
m$residuals(...)
m$log_likelihood(...)
waic(log_likelihood)
dic(log_likelihood)
m$information(..., draw = NULL)
m$reliability(method = "ratio", person_ids = NULL)
```

- Predictive methods return xarray trees. Prior prediction defaults to
  `draws = 500L` and `save_responses = FALSE`. Posterior prediction uses all
  posterior states when `draws = NULL` and includes probabilities when
  `include_probabilities = TRUE`. Both default to `random_seed = NULL` and
  `store = TRUE`.
- Select exactly one `block` or `item` in `predictive_view()`. The R list
  contains observed responses `y`, a matrix with one predictive draw
  per row and one observation per column in `yrep`, supporting response
  categories, and observation and sample coordinates.
- `chi_ppp()` returns an R list with `summary` and labelled discrepancy draws.
  It accepts `predictions = NULL`, `groupby = "overall"`, `draws = NULL`,
  `random_seed = NULL`, `save_discrepancy = FALSE`, `store = TRUE` and
  `progressbar = TRUE`. Pass the result to `plot_ppp()` for an editable ggplot.
- `residuals()` returns a labelled xarray dataset. `log_likelihood()`
  returns a labelled chain × draw × observation array. Both default to
  `progressbar = TRUE`, and log likelihood defaults to `store = TRUE`.
  Pass that array to `waic()` or `dic()` for named R lists of comparison statistics.
- `information()` returns labelled expected Fisher information. By default
  it uses the fitted posterior parameter point, with `parameters = NULL`,
  `theta = NULL`, `include_intermediates = FALSE`, `store = TRUE` and
  `progressbar = TRUE`. Set `draw = c(chain, draw)` to select a single state
  by its original labels. `reliability()` returns a data frame by trait.

## Score, simulate and restore

```r
m$score(new_responses, ..., format = NULL, var_names = NULL)
m$score_model(new_responses, ..., format = NULL)
m$simulate(..., draw = NULL)
save_results(m, path)
m$attach_samples(samples, assume_same_batch = FALSE)
```

- Pass a new response data frame and its `format` to `score()`. It returns a
  fitted R model for the new respondents. For a mixed test, use
  `new_responses = list(fc = fc_answers, items = item_answers)` and
  `format = c(fc = "comparison_long", items = "item_long")`. Pass
  `person_data` with the trained group and predictor columns. Scoring reuses
  the calibration's group labels and fitted formula encoding. Both scoring methods default to
  `calibration = "point"`, which fixes the calibration parameters. Use
  `calibration = "posterior"` for experimental scoring with calibration
  uncertainty. `score_model()` prepares the new model for a separate
  estimation step. `score()` also accepts the fitting options.
- `simulate()` returns a separate unfitted model with simulated responses.
  It defaults to `source = "prior"`. Use `source = "posterior"` to simulate
  from a fitted model, or `source = "fixed"` with explicit `parameters`.
- `save_results()` writes NetCDF and invisibly returns the path. To restore,
  prepare a matching model and pass the file path to its `attach_samples()`
  method. The method also accepts a native result tree.

### Mixed FC and ordinal scoring

For a test with FC blocks and Likert items, register the two channels on the
same model. Supply the calibration people with `person_id`, `school` and
`age`, and use the same columns for new respondents.

```r
m <- model()
m$add_fc(fc_design, fc_answers, format = "comparison_long", model = "TIRT")
m$add_items(item_design, item_answers, format = "item_long", model = "GPCM",
            categories = 0:2)
m$prepare(person_data = calibration_people,
          population = list(group = "school", reference_group = "A",
                            regression_by_group = TRUE))
m$add_regressions("theta ~ age", person_data = calibration_people)
m$fit(random_seed = 42L)

coef(m, blocks = "B1", items = "I1", kind = "stats")
coef(m, component = "regression", kind = "stats")
scored <- m$score(list(fc = new_fc, items = new_items),
                  format = c(fc = "comparison_long", items = "item_long"),
                  person_data = new_people, random_seed = 43L)
```

The [mixed response types example](../tutorials/joint_r.ipynb) shows a fitted model and its coefficient table.

The [TIRT in R example](../tutorials/tirt_r.ipynb) uses the same calibration for
model checking, scoring and restoring saved draws. Python options passed through
`...` follow the [Python API reference](../api.md).
