# PyFCIRT for R

The R interface uses the Python PyFCIRT backend through reticulate.
Use Python 3.12 or later and install the matching checkout with
`python -m pip install ".[io,covariates,numpyro]"`. Set `RETICULATE_PYTHON` to that
environment's Python executable before starting R. This development R package expects the Python API
with `data_coverage()` and `response_counts()`.

## TIRT with triplets

With the matching Python source environment active, run `R CMD INSTALL r/pyfcirt`
from the repository root. Install `bayesplot` to run the diagnostic and scoring plots.

Load the bundled design and pairwise answers. Hold out ten respondents for new-person scoring.

```r
library(pyfcirt)

example <- load_example("tirt_study", model = "TIRT")
head(example$design, 3)
head(example$responses, 3)

answers <- example$responses
new_ids <- tail(unique(answers$person_id), 10L)
is_new <- answers$person_id %in% new_ids

m <- model()
m$add_fc(example$design, answers[!is_new, ], format = example$format,
         model = "TIRT", response_format = example$response_format)
m$prepare()
summary(m)

m$fit(random_seed = 42L)
coef(m, blocks = "B1")

loading <- as_draws(m, variables = "loading")
bayesplot::mcmc_rank_overlay(loading, pars = posterior::variables(loading)[1L])
```

The design uses `block_id`, `position`, `statement_id`, `trait` and `keying`.
The response table uses `person_id`, `block_id`, `statement_1`,
`statement_2` and `response`. A response of 1 favors the first statement;
0 favors the second.

## Check and score responses

Inspect fit for B1 and score the held-out people using the fitted calibration.

```r
check <- m$chi_ppp(blocks = "B1", random_seed = 43L)
plot_ppp(check)

scored <- m$score(answers[is_new, ], format = example$format, random_seed = 44L)
draws <- as_draws(scored, variables = "theta", coords = list(person = new_ids[1]))
posterior::summarise_draws(draws, "mean", "sd")
bayesplot::mcmc_areas_ridges(draws)
```

`as_draws()` returns an iteration-by-chain-by-variable array. Use its
`coordinates` and `sample_coordinates` attributes to inspect the original
parameter and sample labels.

## Save and restore results

```r
save_results(m, "tirt.nc")

restored <- model()
restored$add_fc(example$design, answers[!is_new, ], format = example$format,
                model = "TIRT", response_format = example$response_format)
restored$prepare()
restored$attach_samples("tirt.nc")
coef(restored, blocks = "B1")
```
