library(pyfcirt)

if (identical(Sys.getenv("PYFCIRT_R_INTEGRATION"), "true")) {
  people <- data.frame(person_id = paste0("P", 1:6), school = rep(c("A", "B"), each = 3),
                       age = seq(20, 45, 5))
  design <- data.frame(block_id = "B1", position = 1:3, statement_id = c("S1", "S2", "S3"),
                       trait = c("A", "B", "A"), keying = c(1L, 1L, -1L))
  fc_answers <- data.frame(person_id = rep(people$person_id, each = 3), block_id = "B1",
                           statement_1 = rep(c("S1", "S1", "S2"), 6),
                           statement_2 = rep(c("S2", "S3", "S3"), 6),
                           response = rep(c(1L, 1L, 0L, 0L, 1L, 1L), 3))
  item_design <- data.frame(item_id = c("I1", "I2"), trait = c("A", "B"))
  item_answers <- data.frame(person_id = rep(people$person_id, each = 2),
                             item_id = rep(c("I1", "I2"), 6), response = rep(0:2, 4))

  m <- model()
  m$add_fc(design, fc_answers, format = "comparison_long", model = "TIRT")
  m$add_items(item_design, item_answers, format = "item_long", model = "GPCM", categories = 0:2)
  m$prepare(population = list(group = "school", reference_group = "A", regression_by_group = TRUE),
            person_data = people)
  m$add_regressions("theta ~ age", person_data = people)

  # Use native valid draws as calibration values without running MCMC.
  pred <- m$prior_predictive(draws = 20L, random_seed = 22L, store = FALSE)
  wide <- reticulate::py_eval(
    "__import__('xarray').DataArray([[[1, 2]]], dims=['chain', 'draw', 'person'], coords={'person': [9007199254740992, 9007199254740993]})",
    convert = FALSE
  )
  draw_identity <- pyfcirt:::draw_component(wide, "theta")
  stopifnot(identical(draw_identity$keys$variable,
                      c("theta[9007199254740992]", "theta[9007199254740993]")),
            identical(unlist(draw_identity$coords$person),
                      c("integer:9007199254740992", "integer:9007199254740993")))

  xr <- reticulate::import("xarray", convert = FALSE)
  prior <- reticulate::py_get_item(pred, "prior")$to_dataset()
  m$attach_samples(xr$DataTree$from_dict(list(posterior = prior)))

  measurement <- coef(m, blocks = "B1", items = "I1", kind = "stats")
  regression <- coef(m, component = "regression", kind = "stats")
  one_item <- coef(m, items = "I1", var_names = "loading", kind = "stats")
  stopifnot(identical(one_item$trait, "A"))
  one_group <- coef(m, component = "regression", var_names = "population_mean",
                    coords = list(population_group = "B"), kind = "stats")
  stopifnot(setequal(unique(measurement$unit), c("B1", "I1")),
            all(measurement$channel[measurement$unit == "B1"] == "fc"),
            all(measurement$channel[measurement$unit == "I1"] == "items"),
            setequal(regression$population_group[regression$parameter == "age"], c("A", "B")),
            all(one_group$population_group == "B"), nrow(one_group) == 2L,
            !any(grepl("NaN", capture.output(print(measurement)), fixed = TRUE)))

  select_one <- function(variable, labels) {
    value <- as_draws(m, variables = variable, coords = labels, chain = 0L, draw = 0L)
    stopifnot(inherits(value, "draws_array"), identical(dim(value), c(1L, 1L, 1L)))
    value
  }
  statement <- select_one("loading", list(block = "B1", statement = "S1"))
  item <- select_one("loading", list(item = "I1", trait = "A"))
  pair <- select_one("cutpoint", list(block = "B1", pair = "S1 > S2"))
  select_one("theta", list(person = "P1", trait = "B"))
  stopifnot(!identical(posterior::variables(statement), posterior::variables(item)),
            inherits(try(as_draws(m, variables = "theta", draw = c(0L, 0L)),
                         silent = TRUE), "try-error"))
  stopifnot(identical(attr(pair, "sample_coordinates"), list(chain = 0L, draw = 0L)))
  for (value in list(statement, item, pair)) {
    variable <- names(attr(value, "coordinates"))
    axis <- paste0(variable, "_id")
    id <- attr(value, "coordinates")[[variable]][[axis]]
    native <- select_one(variable, setNames(list(id), axis))
    stopifnot(identical(as.numeric(value), as.numeric(native)))
  }
  stopifnot(inherits(try(as_draws(m, variables = "loading", coords = list(statement = "missing")),
                         silent = TRUE), "try-error"))

  # Submit new people in a different row order from their predictor table.
  new <- data.frame(person_id = c("N2", "N1"), school = c("B", "A"), age = c(55, 60))
  fresh <- list(fc = transform(fc_answers[1:6, ], person_id = rep(c("N1", "N2"), each = 3)),
                items = transform(item_answers[1:4, ], person_id = rep(c("N1", "N2"), each = 2)))
  scored <- m$score_model(fresh, format = c(fc = "comparison_long", items = "item_long"),
                          person_data = new)
  native <- attr(scored, "python")
  population <- native$prepared$population
  centered <- reticulate::py_to_r(population$predictors)
  group <- reticulate::py_to_r(population$person_group)
  expected <- new$age[match(c("N1", "N2"), new$person_id)] - mean(people$age[people$school == "A"])
  graph <- native$build()
  stopifnot(all(scored$data_coverage()$n_persons == 2L),
            identical(as.numeric(centered), expected), identical(as.integer(group), 0:1),
            is.finite(reticulate::py_to_r(graph$compile_logp()(graph$initial_point()))))
}
