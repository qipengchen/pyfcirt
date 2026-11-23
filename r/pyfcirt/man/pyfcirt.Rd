\name{pyfcirt}
\alias{pyfcirt}
\alias{model}
\alias{load_example}
\alias{summary.pyfcirt_model}
\alias{coef.pyfcirt_model}
\alias{response_counts}
\alias{predictive_view}
\alias{plot_ppp}
\alias{as_draws}
\alias{waic}
\alias{dic}
\alias{save_results}
\title{Fit and analyze PyFCIRT models from R}
\description{
Use a single R model object to prepare designs, fit the Python model, and
analyze its results. See the package README for the TIRT walkthrough.
}
\usage{
model()
load_example(name, model, n_persons = NULL, random_seed = NULL, include_truth = FALSE)
\method{summary}{pyfcirt_model}(object, details = FALSE, ...)
\method{coef}{pyfcirt_model}(object, blocks = NULL, items = NULL,
  component = "measurement", ci_prob = 0.95, ci_kind = "hdi", var_names = NULL,
  coords = list(), kind = "all", digits = 3L, ...)
response_counts(answers, groupby = NULL)
predictive_view(predictions, block = NULL, item = NULL)
plot_ppp(check, coords = list(), bins = 30L, quantile_range = NULL, max_panels = 12L)
as_draws(m, variables = NULL, coords = list(), chain = NULL, draw = NULL)
waic(log_likelihood)
dic(log_likelihood)
save_results(m, path)
}
\arguments{
\item{name}{Bundled example ID.}
\item{model}{Measurement model name, specification, or named channel list.}
\item{n_persons,random_seed,include_truth}{Optional cohort size, seed, and whether to read simulation truth.}
\item{object,m}{An R model created by \code{model()}.}
\item{details}{Include expanded model details.}
\item{blocks,items}{IDs for units to include.}
\item{component}{Measurement or regression coefficients.}
\item{ci_prob,ci_kind}{Credible interval probability and interval type.}
\item{var_names,variables}{Names of canonical parameters.}
\item{coords}{For \code{as_draws()}, a named list of dimension or entity labels.
Use block, statement, pair, item or trait labels. Pair labels match
\code{coef()} display, for example \code{"S1 > S2"}. Fields combine by
intersection and variables without matches are omitted.}
\item{kind}{Type of coefficient summary.}
\item{digits}{Digits shown by the print method.}
\item{...}{Additional summary options passed to Python.}
\item{answers,groupby}{Observed response table and optional additional grouping columns.}
\item{predictions}{A prior or posterior predictive result from the model.}
\item{block,item}{ID of one unit to select. Supply exactly one.}
\item{check}{A result from \code{m$chi_ppp()}.}
\item{bins,quantile_range,max_panels}{Display controls for the PPP histogram.}
\item{chain,draw}{Original sample labels. Leave NULL to retain all labels.}
\item{log_likelihood}{Pointwise log likelihood from \code{m$log_likelihood()}.}
\item{path}{Path of the NetCDF results file.}
}
\value{
\code{model()} returns a model object. \code{load_example()} and
\code{predictive_view()} return lists. \code{summary()} returns a printable
model description. \code{coef()} and \code{response_counts()} return data
frames. \code{as_draws()} returns a posterior draws array with coordinate
attributes and stable names based on the original dimension labels.
\code{plot_ppp()} returns a ggplot. \code{waic()} and \code{dic()}
return named lists. \code{save_results()} invisibly returns the path.
}
\details{
Register data using \code{m$add_fc()} or \code{m$add_items()} and call
\code{m$prepare()} before fitting. For groups, supply \code{person_data} and
a population list such as \code{list(group = "school", reference_group = "A")}.
Use \code{m$data_coverage()} to check
administration counts, or \code{m$response_data()} to extract observed rows.
Use \code{m$set_population()} before preparation to register the group column,
reference group and respondent table. After preparation, use
\code{m$set_priors()}, \code{m$add_regressions()}, or the experimental
\code{m$add_dif_terms()} before \code{m$fit()}. The DIF interface accepts
a character vector of targets or a named list of target options.
\code{m$available_covariate_targets()} and \code{m$dif_summary()} return
data frames with target or contrast labels.

Obtain predictive draws with \code{m$prior_predictive()} and
\code{m$posterior_predictive()}. The fitted model also provides
\code{m$chi_ppp()}, \code{m$residuals()}, \code{m$information()},
\code{m$reliability()}, and \code{m$trait_summary()}. Compare candidates only
when their observation units and observations match.

\code{m$score()} estimates a new-respondent model.
\code{m$score_model()} prepares that model for a chosen estimator. Both
accept a named list of new response data frames and a named \code{format}
vector for mixed tests, with channels \code{fc} and \code{items}.
\code{m$simulate()} returns a separate unfitted model with simulated responses.
Rebuild the matching prepared model before using \code{m$attach_samples()}
to restore saved draws.
}
