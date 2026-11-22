library(pyfcirt)

stopifnot(!reticulate::py_available(initialize = FALSE))

if (identical(Sys.getenv("PYFCIRT_R_INTEGRATION"), "true")) {
  example <- load_example("tirt_study", model = "TIRT")
  answers <- example$responses
  new_ids <- tail(unique(answers$person_id), 10L)
  is_new <- answers$person_id %in% new_ids

  prepare_calibration <- function() {
    m <- model()
    m$add_fc(example$design, answers[!is_new, ], format = example$format, model = "TIRT",
             response_format = example$response_format)
    m$prepare()
    m
  }

  m <- prepare_calibration()
  stopifnot(m$data_coverage()$n_persons == 90L)
  observed <- m$response_data(blocks = "B1")
  counts <- response_counts(observed)
  stopifnot(sum(counts$count) == nrow(observed), all(counts$proportion >= 0))

  prior <- m$prior_predictive(draws = 3L, random_seed = 30L, store = FALSE)
  view <- predictive_view(prior, block = "B1")
  stopifnot(nrow(view$yrep) == 3L, ncol(view$yrep) == length(view$y),
            nrow(view$samples) == 3L, length(view$response_support) > 0L)

  if (identical(Sys.getenv("PYFCIRT_R_SAMPLE"), "true")) {
    sample_args <- list(draws = 40L, tune = 40L, chains = 2L, progressbar = FALSE,
                        nuts_sampler = "numpyro", nuts = list(chain_method = "vectorized"))
    do.call(m$fit, c(sample_args, list(random_seed = 42L)))
    coefficients <- coef(m, blocks = "B1", kind = "stats")
    stopifnot(nrow(coefficients) > 0L, is.numeric(coefficients$mean))
    predicted <- m$posterior_predictive(draws = 3L, random_seed = 43L, store = FALSE,
                                       progressbar = FALSE)
    view <- predictive_view(predicted, block = "B1")
    stopifnot(nrow(view$yrep) == 3L, "source_draw" %in% names(view$samples))
    check <- m$chi_ppp(predictions = predicted, blocks = "B1", store = FALSE,
                       progressbar = FALSE)
    stopifnot(check$summary$PPP >= 0, check$summary$PPP <= 1,
              inherits(plot_ppp(check), "ggplot"))
    logs <- m$log_likelihood(store = FALSE, progressbar = FALSE)
    stopifnot(is.finite(waic(logs)$WAIC), is.finite(dic(logs)$DIC))

    path <- tempfile(fileext = ".nc")
    save_results(m, path)
    restored <- prepare_calibration()
    restored$attach_samples(path)
    stopifnot(identical(as_draws(m, variables = "loading"),
                        as_draws(restored, variables = "loading")))
    unlink(path)

    scored <- do.call(restored$score, c(list(new_responses = answers[is_new, ],
                                             format = example$format, random_seed = 44L),
                                        sample_args))
    theta <- as_draws(scored, variables = "theta", coords = list(person = new_ids[1L]),
                      chain = 0L, draw = 0L)
    stopifnot(identical(dim(theta)[1:2], c(1L, 1L)),
              all(attr(theta, "coordinates")$theta$person == new_ids[1L]),
              scored$data_coverage()$n_persons == 10L)
  }
}
