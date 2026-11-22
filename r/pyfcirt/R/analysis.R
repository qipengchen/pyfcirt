predictive_view <- function(predictions, block = NULL, item = NULL) {
  view <- backend()$predictive_view(predictions, block = block, item = item)
  observed <- reticulate::py_get_item(view, "observed_data")$to_dataset()
  role <- if ("posterior_predictive" %in% python_names(view$children$keys())) {
    "posterior_predictive"
  } else {
    "prior_predictive"
  }
  replicated <- reticulate::py_get_item(view, role)$to_dataset()
  obs <- reticulate::py_get_item(observed, "response")
  response <- reticulate::py_get_item(replicated, "response")$transpose("chain", "draw", "obs")
  values <- to_r(response$values)
  n <- dim(values)
  samples <- expand.grid(chain = axis_labels(response, "chain"), draw = axis_labels(response, "draw"),
                         KEEP.OUT.ATTRS = FALSE)
  coordinates <- python_names(response$coords$keys())
  for (axis in intersect(c("source_chain", "source_draw"), coordinates)) {
    samples[[axis]] <- rep(axis_labels(response, axis), each = n[[1L]])
  }
  support <- reticulate::py_get_item(view, "response_support")$to_dataset()
  list(y = to_r(obs$values), yrep = matrix(values, n[1] * n[2], n[3]),
       obs = dataset_table(observed),
       samples = samples, response_support = dataset_tables(support),
       response_kind = to_r(reticulate::py_get_item(view$attrs, "response_kind")))
}

waic <- function(log_likelihood) dataset_tables(backend()$waic(log_likelihood))

dic <- function(log_likelihood) dataset_tables(backend()$dic(log_likelihood))

plot_ppp <- function(check, coords = list(), bins = 30L, quantile_range = NULL, max_panels = 12L) {
  check <- select_labels(check$draws, coords)
  difference <- check$D_diff
  axes <- setdiff(unlist(to_r(difference$dims), use.names = FALSE), c("chain", "draw"))
  table <- as_table(difference$to_dataframe(name = "D_diff"))
  if (any(!is.finite(table$D_diff))) {
    stop("Discrepancies exceed the plotting range.", call. = FALSE)
  }
  panels <- if (length(axes)) table[[axes[[1L]]]] else rep("Overall", nrow(table))
  if (length(unique(panels)) > max_panels) stop("Select fewer panels with coords.", call. = FALSE)
  table$panel <- panels
  if (!is.null(quantile_range)) {
    table <- do.call(rbind, lapply(split(table, table$panel), function(part) {
      limits <- stats::quantile(part$D_diff, quantile_range)
      part[part$D_diff >= limits[[1L]] & part$D_diff <= limits[[2L]], , drop = FALSE]
    }))
  }
  ggplot2::ggplot(table, ggplot2::aes(x = !!as.name("D_diff"))) +
    ggplot2::geom_histogram(bins = bins) +
    ggplot2::geom_vline(xintercept = 0) +
    ggplot2::facet_wrap(stats::as.formula("~ panel")) +
    ggplot2::labs(x = "Replicated minus observed discrepancy", y = "Draws")
}
