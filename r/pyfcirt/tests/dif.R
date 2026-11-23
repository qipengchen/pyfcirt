
library(pyfcirt)

if (identical(Sys.getenv("PYFCIRT_R_INTEGRATION"), "true")) {
  example <- load_example("tirt_dif", model = "TIRT", n_persons = 16L)
  people <- example$person_data
  people$language <- ave(seq_len(nrow(people)), people$site, FUN = seq_along) %% 2L
  people$language <- ifelse(people$language == 0L, "en", "zh")
  prepare_dif <- function() {
    m <- model()
    m$add_fc(example$design, example$responses, format = example$format,
             model = "TIRT", response_format = example$response_format)
    m$set_population(group = "site", reference_group = "A", person_data = people)
    m$prepare()
    m
  }

  m <- prepare_dif()
  eligible <- m$available_covariate_targets(kind = "dif")
  stopifnot(is.data.frame(eligible),
            setequal(eligible$target, c("fc.cutpoints", "fc.loading_magnitude")))
  m$add_dif_terms(
    list(
      "fc.loading_magnitude" = list(anchors = c("B01_1", "B01_2")),
      "fc.cutpoints" = list(
        anchors = data.frame(block_id = "B01", statement_1 = "B01_2", statement_2 = "B01_1"),
        prior_options = list(b = 0.25)
      )
    ),
    formula = "site * language", reference_levels = list(language = "en")
  )
  text <- as.character(summary(m))
  stopifnot(grepl("Reference levels: {'site': 'A', 'language': 'en'}", text, fixed = TRUE),
            grepl("DIF anchors (cutpoints ~ site * language): (('B01', 'B01_1', 'B01_2'),)",
                  text, fixed = TRUE))

  # Supply a known point at two draw labels to test conversion without sampling.
  fixture <- reticulate::py_run_string(paste(
    "import numpy as np",
    "import xarray as xr",
    "def dif_point(model):",
    "    prepared = model.prepared",
    "    values = {}",
    "    for name, layout in prepared.point_layouts.items():",
    "        if name in ('covariance', 'trait_correlation'):",
    "            value = np.broadcast_to(np.eye(layout.shape[-1]), layout.shape)",
    "        else:",
    "            free = 1.0 if name in ('loading', 'population_sd') else 0.0",
    "            value = np.where(np.isnan(layout.fixed), free, layout.fixed)",
    "        values[name] = (layout.dims, value)",
    "    values['theta'] = (('person', 'trait'), np.zeros((16, 3)))",
    "    beta = values['regression_coefficient'][1]",
    "    for plan in prepared.person_effect_regressions:",
    "        terms = list(plan.design.term.values)",
    "        if 'site[B]' in terms:",
    "            beta[plan.coefficient_entries[:, terms.index('site[B]')]] = 0.25",
    "            beta[plan.coefficient_entries[:, terms.index('site[B]:language[zh]')]] = 0.5",
    "    dims = {dim for names, _ in values.values() for dim in names}",
    "    point = xr.Dataset(values, coords={dim: prepared.coords[dim] for dim in dims})",
    "    posterior = prepared.annotate_coordinates(point.expand_dims(chain=[0], draw=[1, 2]))",
    "    return xr.DataTree.from_dict({'posterior': posterior})",
    sep = "\n"
  ), convert = FALSE)
  native <- attr(m, "python")
  m$attach_samples(fixture$dif_point(native))
  for (lang in c("en", "zh")) {
    result <- m$dif_summary(contrasts = list(site = c("A", "B")), at = list(language = lang))
    expected <- if (lang == "en") 0.25 else 0.75
    stopifnot(is.data.frame(result), nrow(result) == 33L,
              all(result$reference == "A"), all(result$compared == "B"),
              !any(unlist(result$units) == "B01"),
              any(result$pair %in% "B02_1 > B02_2"),
              all(abs(result$mean[result$parameter == "cutpoint"] - expected) < 1e-12),
              all(abs(result$mean[result$parameter == "loading"] - expm1(expected)) < 1e-12),
              identical(result$mean, result$lower), identical(result$mean, result$upper))
  }

  joint <- prepare_dif()
  joint$add_dif_terms(
    "fc.cutpoints", groups = list(c("site", "language")), prior = "pairwise_l1"
  )
  stopifnot(grepl("DIF: cutpoint ~ site:language",
                  as.character(summary(joint)), fixed = TRUE))
  joint$attach_samples(fixture$dif_point(attr(joint, "python")))
  crossed <- joint$dif_summary(contrasts = list(
    "site:language" = list(c("A", "en"), c("B", "en"))
  ))
  stopifnot(nrow(crossed) == 12L, all(crossed$mean == 0),
            identical(unlist(crossed$reference[[1L]]), c("A", "en")),
            identical(unlist(crossed$compared[[1L]]), c("B", "en")))
}
