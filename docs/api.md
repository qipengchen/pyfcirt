# API reference

## Workflow overview

Start at the stage that matches your task. For a first fit, follow the
[Quick start](guide/getting_started.md).

### 1. Prepare data

| API | Purpose |
| --- | --- |
| [`fc.Model()`](reference/generated/pyfcirt.Model.md) | Start an analysis. |
| [`fc.load_example()`](reference/generated/pyfcirt.load_example.md) | Load bundled data for an example or practice analysis. |
| [`model.add_fc()`](reference/generated/pyfcirt.Model.md#add-a-forced-choice-response-table) | Register a forced-choice design and its responses. |
| [`model.add_items()`](reference/generated/pyfcirt.Model.md#add-an-individual-item-response-table) | Register a design and responses for binary or rating items. |
| [`model.prepare()`](reference/generated/pyfcirt.Model.md#prepare-data) | Validate the design and responses and prepare the model for fitting. |
| [`model.data_summary()`](reference/generated/pyfcirt.data_summary.md) | Check numbers of respondents, administered units, answers, and missing responses. |
| [`model.response_data()`](reference/generated/pyfcirt.response_data.md) | Inspect observed response records. |
| [`fc.response_summary()`](reference/generated/pyfcirt.response_summary.md) | Summarize counts or proportions for observed response categories. |
| [`fc.plot_response_distribution()`](reference/generated/pyfcirt.plot_response_distribution.md) | Plot response counts or proportions. |

### 2. Configure and fit a model

| API | Purpose |
| --- | --- |
| [`model.summary()`](reference/generated/pyfcirt.Model.summary.md) | Review the prepared model, its data, and its parameter settings. |
| [`model.set_priors()`](reference/generated/pyfcirt.Model.set_priors.md) | Override selected default priors. |
| [`fc.PopulationSpec()`](reference/generated/pyfcirt.PopulationSpec.md) | Configure the trait population, including group structure. |
| [`model.add_regressions()`](reference/generated/pyfcirt.Model.add_regressions.md) | Add covariates to trait scores or measurement parameters. |
| [`model.prior_predictive()`](reference/generated/pyfcirt.prior_predictive.md) | Check which response patterns the priors imply before fitting. |
| [`model.fit()`](reference/generated/pyfcirt.Model.fit.md) | Fit the model and obtain posterior draws. |

### 3. Check sampling and model fit

| API | Purpose |
| --- | --- |
| [`az.summary(..., kind="diagnostics")`](https://python.arviz.org/en/stable/api/generated/arviz.summary.html) | Check R-hat and effective sample sizes using `model.idata`. |
| [`azp.plot_rank_dist()`](https://python.arviz.org/projects/plots/en/stable/api/generated/arviz_plots.plot_rank_dist.html) | Plot posterior distributions and ranks across chains using `model.idata`. |
| [`model.posterior_predictive()`](reference/generated/pyfcirt.posterior_predictive.md) | Generate replicated responses to compare with observed answers. |
| [`fc.predictive_view()`](reference/generated/pyfcirt.predictive_view.md) | Select observed and replicated responses for one block or item. |
| [`model.chi_ppp()`](reference/generated/pyfcirt.chi_ppp.md) | Run a posterior predictive check for categorical responses using a Pearson discrepancy. |
| [`fc.plot_ppp()`](reference/generated/pyfcirt.plot_ppp.md) | Plot results of the posterior predictive check. |
| [`model.residuals()`](reference/generated/pyfcirt.residuals.md) | Calculate observed-minus-expected responses to locate discrepancies. |
| [`model.log_likelihood()`](reference/generated/pyfcirt.log_likelihood.md) | Obtain pointwise log likelihoods for the observed responses. |
| [`fc.waic()`](reference/generated/pyfcirt.waic.md) | Compare candidate models fitted to the same observations and observation units. |

### 4. Interpret and report results

| API | Purpose |
| --- | --- |
| [`model.coef()`](reference/generated/pyfcirt.coef.md) | Summarize measurement parameters and their posterior uncertainty. |
| [`model.trait_summary()`](reference/generated/pyfcirt.trait_summary.md) | Summarize population trait means, standard deviations, and correlations. |
| [`model.idata.posterior["theta"]`](reference/results.md#inspect-or-export-results) | Access the posterior draws for each respondent's trait scores. |
| [`fc.plot_ridge()`](reference/generated/pyfcirt.plot_ridge.md) | Plot the posterior distributions of selected trait scores. |
| [`model.information()`](reference/generated/pyfcirt.information.md) | Evaluate how much measurement information the responses provide at a trait level. |
| [`model.reliability()`](reference/generated/pyfcirt.reliability.md) | Estimate score reliability from posterior trait scores and their uncertainty. |

### 5. Score new respondents

| API | Purpose |
| --- | --- |
| [`model.score()`](reference/generated/pyfcirt.score.md) | Estimate trait scores for new respondents using an existing calibration. Experimental mode `calibration="posterior"` lets new responses update shared calibration parameters. |
| [`model.score_model()`](reference/generated/pyfcirt.score_model.md) | Build the new-respondent model for inspection or fitting with a chosen estimator. |
| [`model.posterior_approximation()`](reference/generated/pyfcirt.Model.posterior_approximation.md) | Fit a runtime prior for experimental posterior scoring and recursive calibration. By default, fit one group across the shared RVs and select its family with `auto`. |

## Full API index

```{toctree}
:maxdepth: 1

reference/data
reference/parameter_contract
reference/measurement
reference/distributions
reference/construction
reference/population
reference/summaries
reference/prediction
reference/information
reference/scoring
reference/posterior_approximation
reference/simulation
reference/results
reference/troubleshooting
```

## Save and reuse results

| API | Purpose |
| --- | --- |
| [`model.idata.to_netcdf()`](reference/results.md#save-and-restore-samples) | Save posterior draws and stored analysis outputs. |
| [`model.attach_samples()`](reference/generated/pyfcirt.Model.attach_samples.md) | Attach saved posterior draws to a matching prepared model. |
