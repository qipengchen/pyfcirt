.onLoad <- function(libname, pkgname) {
  reticulate::py_require(python_version = ">=3.12")
}

backend <- function() {
  module <- reticulate::import("pyfcirt", convert = FALSE)
  if (!reticulate::py_has_attr(module, "data_coverage")) {
    stop("Install PyFCIRT from the matching checkout in the Python environment used by R.",
         call. = FALSE)
  }
  module
}

to_r <- reticulate::py_to_r

model <- function() wrap_model(backend()$Model())

wrap_model <- function(py) {
  m <- structure(list(
    add_fc = function(design, responses, format, model, response_format = "rank",
                      person_id_column = NULL, ...) {
      py$add_fc(design, responses, format = format, model = model,
                response_format = response_format, person_id_column = person_id_column, ...)
      invisible(m)
    },
    add_items = function(design, responses, format, model, kind = NULL,
                         categories = c(0L, 1L), person_id_column = NULL, ...) {
      if (!is.null(kind)) {
        if (!identical(model, "DichotomousIRT")) {
          stop("kind is only available for DichotomousIRT.", call. = FALSE)
        }
        model <- backend()$DichotomousIRT(kind = kind)
      }
      py$add_items(design, responses, format = format, model = model,
                   categories = as.list(categories), person_id_column = person_id_column, ...)
      invisible(m)
    },
    set_population = function(group = NULL, reference_group = NULL, person_data = NULL, ...) {
      py$set_population(group = group, reference_group = reference_group,
                        person_data = person_data, ...)
      invisible(m)
    },
    prepare = function(..., population = NULL) {
      if (is.list(population)) {
        population <- do.call(backend()$PopulationSpec, population)
      }
      do.call(py$prepare, c(list(...), if (!is.null(population)) list(population = population)))
      invisible(m)
    },
    set_priors = function(priors) {
      py$set_priors(priors)
      invisible(m)
    },
    add_regressions = function(formulas, ...) {
      py$add_regressions(formulas, ...)
      invisible(m)
    },
    available_covariate_targets = function(level = NULL, kind = NULL) {
      as_table(py$available_covariate_targets(level = level, kind = kind))
    },
    add_dif_terms = function(targets, formula = NULL, ...) {
      options <- list(...)
      if (is.list(options$groups)) {
        options$groups <- lapply(options$groups, sample_label)
      }
      do.call(py$add_dif_terms, c(list(targets = dif_targets(targets), formula = formula), options))
      invisible(m)
    },
    dif_summary = function(contrasts = NULL, at = NULL, prob = 0.95, significant_only = FALSE) {
      if (!is.null(contrasts)) {
        contrasts <- lapply(contrasts, function(endpoints) {
          endpoints <- lapply(endpoints, function(profile) {
            if (length(profile) > 1L) sample_label(profile) else profile
          })
          sample_label(endpoints)
        })
      }
      as_table(py$dif_summary(contrasts = contrasts, at = at, prob = prob,
                             significant_only = significant_only))
    },
    fit = function(..., var_names = NULL) {
      py$fit(..., var_names = selection(var_names))
      invisible(m)
    },
    data_coverage = function(by = "model", details = FALSE) {
      as_table(py$data_coverage(by = by, details = details))
    },
    response_data = function(blocks = NULL, items = NULL, representation = "auto", groups = NULL) {
      if (!is.null(groups)) {
        groups <- reticulate::r_to_py(groups, convert = FALSE)$set_index("person")
      }
      as_table(py$response_data(blocks = selection(blocks), items = selection(items),
                                representation = representation, groups = groups))
    },
    trait_summary = function(by_group = FALSE, ci_prob = 0.95, ci_kind = "hdi", digits = 3L, ...) {
      result <- py$trait_summary(by_group = by_group, ci_prob = ci_prob, ci_kind = ci_kind, ...)
      structure(dataset_tables(result), class = "pyfcirt_traits", digits = digits)
    },
    prior_predictive = function(...) py$prior_predictive(...),
    posterior_predictive = function(...) py$posterior_predictive(...),
    chi_ppp = function(blocks = NULL, items = NULL, ...) {
      result <- py$chi_ppp(blocks = selection(blocks), items = selection(items), ...)
      summary <- reticulate::py_get_item(result, list("PPP", "PPP_mcse", "n_obs"))
      list(summary = dataset_tables(summary), draws = result)
    },
    residuals = function(...) py$residuals(...),
    information = function(..., draw = NULL) py$information(..., draw = sample_label(draw)),
    reliability = function(method = "ratio", person_ids = NULL) {
      dataset_table(py$reliability(method = method, person_ids = selection(person_ids)))
    },
    log_likelihood = function(...) py$log_likelihood(...),
    score = function(new_responses, ..., format = NULL, var_names = NULL) {
      wrap_model(do.call(py$score, c(scoring_inputs(new_responses, format),
                                    list(..., var_names = selection(var_names)))))
    },
    score_model = function(new_responses, ..., format = NULL) {
      wrap_model(do.call(py$score_model, c(scoring_inputs(new_responses, format), list(...))))
    },
    simulate = function(..., draw = NULL) wrap_model(py$simulate(..., draw = sample_label(draw))),
    attach_samples = function(samples, assume_same_batch = FALSE) {
      if (is.character(samples)) {
        xr <- reticulate::import("xarray", convert = FALSE)
        tree <- xr$open_datatree(path.expand(samples), engine = "h5netcdf")
        on.exit(tree$close())
        samples <- tree$load()
      }
      py$attach_samples(samples, assume_same_batch = assume_same_batch)
      invisible(m)
    }
  ), class = "pyfcirt_model", python = py)
  m
}

scoring_inputs <- function(responses, format) {
  if (is.list(responses) && !is.data.frame(responses)) {
    channels <- names(responses)
    if (is.null(channels) || anyDuplicated(channels) ||
        !all(channels %in% c("fc", "items")) || !setequal(channels, names(format))) {
      stop("Name the response tables and their formats fc and/or items.", call. = FALSE)
    }
    response_table <- backend()$ResponseTable
    responses <- stats::setNames(lapply(channels, function(channel) {
      response_table(responses[[channel]], format = format[[channel]])
    }), channels)
    format <- NULL
  }
  list(new_responses = responses, format = format)
}

dif_targets <- function(targets) {
  if (!is.list(targets)) return(selection(targets))
  lapply(targets, function(config) {
    if (!length(config)) return(reticulate::dict())
    anchors <- config$anchors
    if (is.data.frame(anchors)) {
      keys <- anchors[c("block_id", "statement_1", "statement_2")]
      keys[] <- lapply(keys, function(column) if (is.factor(column)) as.character(column) else column)
      config$anchors <- lapply(seq_len(nrow(keys)), function(i) {
        sample_label(unname(as.list(keys[i, , drop = FALSE])))
      })
    } else if (!is.null(anchors)) {
      config$anchors <- selection(anchors)
    }
    config
  })
}

selection <- function(ids) if (is.null(ids)) NULL else as.list(ids)

sample_label <- function(draw) {
  if (is.null(draw)) NULL else do.call(reticulate::tuple, as.list(draw))
}

python_model <- function(m) attr(m, "python", exact = TRUE)

save_results <- function(m, path) {
  python_model(m)$idata$to_netcdf(path.expand(path), engine = "h5netcdf")
  invisible(path)
}

load_example <- function(name, model, n_persons = NULL, random_seed = NULL, include_truth = FALSE) {
  example <- backend()$load_example(name, model = model, n_persons = n_persons,
                                   random_seed = random_seed, include_truth = include_truth)
  input_tables <- function(input) {
    result <- list(design = as_table(input$design), responses = as_table(input$responses$frame),
                   format = to_r(input$responses$format))
    if (reticulate::py_has_attr(input, "response_format")) {
      result["response_format"] <- list(to_r(input$response_format))
      result["scale"] <- list(input$scale)
    } else {
      result$categories <- to_r(input$categories)
    }
    result
  }
  data <- example$data
  if (reticulate::py_has_attr(data, "design")) {
    inputs <- input_tables(data)
  } else {
    builtins <- reticulate::import_builtins(convert = FALSE)
    inputs <- list()
    for (i in seq_len(to_r(builtins$len(data)))) {
      input <- reticulate::py_get_item(data, i - 1L)
      channel <- if (reticulate::py_has_attr(input, "response_format")) "fc" else "items"
      inputs[[channel]] <- input_tables(input)
    }
  }
  c(inputs, list(person_data = to_r(example$person_data), metadata = to_r(example$metadata),
                 truth = to_r(example$truth)))
}
