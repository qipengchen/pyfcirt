# API reference

For a first fit, follow the [Quick start](guide/getting_started.md).
For R, see the [R API reference](reference/r_api.md).

(workflow-overview)=
## Prepare data

| API | Purpose |
| --- | --- |
| [`fc.Model()`](reference/generated/pyfcirt.Model.md) | Start an analysis. |
| [`fc.load_example()`](reference/generated/pyfcirt.load_example.md) | Load bundled data for an example or practice analysis. |
| [`model.add_fc()`](reference/generated/pyfcirt.Model.md#add-a-forced-choice-response-table) | Register a forced-choice design and its responses. |
| [`model.add_items()`](reference/generated/pyfcirt.Model.md#add-an-individual-item-response-table) | Register a design and responses for binary or rating items. |
| [`model.prepare()`](reference/generated/pyfcirt.Model.md#prepare-data) | Validate the design and responses and prepare the model for fitting. |
| [`model.data_coverage()`](reference/generated/pyfcirt.data_coverage.md) | Check numbers of respondents, administered units, answers, and missing responses. |
| [`model.response_data()`](reference/generated/pyfcirt.response_data.md) | Inspect observed response records. |

See [observed response distributions](reference/summaries.md#observed-response-distributions)
for counts, proportions, and plots.

## Configure and fit a model

| API | Purpose |
| --- | --- |
| [`model.summary()`](reference/generated/pyfcirt.Model.summary.md) | Review the prepared model, its data, and its parameter settings. |
| [`model.set_priors()`](reference/generated/pyfcirt.Model.set_priors.md) | Override selected default priors. |
| [`model.add_regressions()`](reference/generated/pyfcirt.Model.add_regressions.md) | Add covariates to trait scores or measurement parameters. |
| [`model.add_dif_terms()`](reference/dif.md#add-dif-terms) | Add respondent predictors to selected measurement parameters. |
| [`model.prior_predictive()`](reference/generated/pyfcirt.prior_predictive.md) | Check which response patterns the priors imply before fitting. |
| [`model.fit()`](reference/generated/pyfcirt.Model.fit.md) | Fit the model and obtain posterior draws. |

## Check sampling and model fit

| API | Purpose |
| --- | --- |
| [`model.posterior_predictive()`](reference/generated/pyfcirt.posterior_predictive.md) | Generate replicated responses to compare with observed answers. |
| [`fc.predictive_view()`](reference/generated/pyfcirt.predictive_view.md) | Select observed and replicated answers for one block or item. |
| [`model.chi_ppp()`](reference/generated/pyfcirt.chi_ppp.md) | Run a posterior predictive check for categorical responses using a Pearson discrepancy. |
| [`fc.plot_ppp()`](reference/generated/pyfcirt.plot_ppp.md) | Plot results of the posterior predictive check. |
| [`model.residuals()`](reference/generated/pyfcirt.residuals.md) | Calculate observed-minus-expected responses to locate discrepancies. |

For convergence diagnostics, use [ArviZ summary](https://python.arviz.org/en/stable/api/generated/arviz.summary.html)
or [rank and distribution plots](https://python.arviz.org/projects/plots/en/stable/api/generated/arviz_plots.plot_rank_dist.html)
with `model.idata`.

## Compare models

| API | Purpose |
| --- | --- |
| [`model.log_likelihood()`](reference/generated/pyfcirt.log_likelihood.md) | Obtain pointwise log likelihoods for the observed responses. |
| [`fc.waic()`](reference/generated/pyfcirt.waic.md) | Estimate expected predictive accuracy from pointwise log likelihoods. |
| [`fc.dic()`](reference/generated/pyfcirt.dic.md) | Compute the deviance information criterion from pointwise log likelihoods. |

Use the same observations and likelihood units when comparing models.

## Interpret and report results

| API | Purpose |
| --- | --- |
| [`model.coef()`](reference/generated/pyfcirt.coef.md) | Summarize measurement parameters and their posterior uncertainty. |
| [`model.dif_summary()`](reference/dif.md#summarize-dif) | Summarize posterior DIF contrasts between respondent profiles. |
| [`model.trait_summary()`](reference/generated/pyfcirt.trait_summary.md) | Summarize population trait means, standard deviations, and correlations. |
| [`model.idata.posterior["theta"]`](reference/results.md#inspect-or-export-results) | Access the posterior draws for each respondent's trait scores. |
| [`fc.plot_ridge()`](reference/generated/pyfcirt.plot_ridge.md) | Plot the posterior distributions of selected trait scores. |
| [`model.information()`](reference/generated/pyfcirt.information.md) | Evaluate how much measurement information the responses provide at a trait level. |
| [`model.reliability()`](reference/generated/pyfcirt.reliability.md) | Estimate score reliability from posterior trait scores and their uncertainty. |

## Score new respondents or simulate data

| API | Purpose |
| --- | --- |
| [`model.score()`](reference/generated/pyfcirt.score.md) | Estimate trait scores for new respondents using an existing calibration. |
| [`model.score_model()`](reference/generated/pyfcirt.score_model.md) | Build a new-respondent scoring model for inspection or a chosen estimator. |
| [`model.simulate()`](reference/generated/pyfcirt.simulate.md) | Generate responses using the current design and specified parameter source. |

## Save and reuse results

| API | Purpose |
| --- | --- |
| [`model.idata.to_netcdf()`](reference/results.md#save-and-restore-samples) | Save posterior draws and stored analysis outputs. |
| [`model.attach_samples()`](reference/generated/pyfcirt.Model.attach_samples.md) | Attach saved posterior draws to a matching prepared model. |

## Advanced modeling

- Use [population and covariate specifications](reference/population.md) for groups and regression formulas.
- Use [`model.build()`](reference/generated/pyfcirt.Model.build.md) and [measurement distributions](reference/distributions.md) to work with native PyMC models.
- See [posterior approximation](reference/posterior_approximation.md) for experimental scoring with calibration uncertainty and recursive updates.

## Developer reference

The [parameter and coordinate contract](reference/parameter_contract.md) describes
prepared layouts, parameter identities and stored coordinates for integrations.

See [troubleshooting](reference/troubleshooting.md) for installation and runtime errors.

<span id="full-api-index"></span>

```{toctree}
:maxdepth: 1
:hidden:

reference/r_api
reference/data
reference/measurement
reference/construction
reference/summaries
reference/prediction
reference/information
reference/scoring
reference/simulation
reference/results
reference/population
reference/dif
reference/distributions
reference/posterior_approximation
reference/parameter_contract
reference/troubleshooting
```
