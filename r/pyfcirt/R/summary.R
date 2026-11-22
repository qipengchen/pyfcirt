as_table <- function(table) {
  if (length(python_names(table$index$names))) {
    table <- table$reset_index()
  }
  list2DF(to_r(table$to_dict(orient = "list")))
}

python_names <- function(keys) {
  builtins <- reticulate::import_builtins(convert = FALSE)
  unlist(to_r(builtins$list(keys)), use.names = FALSE)
}

dataset_table <- function(dataset) as_table(dataset$to_dataframe())

dataset_tables <- function(dataset) {
  variables <- python_names(dataset$data_vars$keys())
  tables <- lapply(variables, function(name) {
    value <- reticulate::py_get_item(dataset, name)
    if (length(to_r(value$dims)) == 0L) return(to_r(value$item()))
    as_table(value$to_dataframe(name = name))
  })
  stats::setNames(tables, variables)
}

summary.pyfcirt_model <- function(object, details = FALSE, ...) {
  structure(to_r(python_model(object)$summary(details = details)), class = "pyfcirt_summary")
}

print.pyfcirt_summary <- function(x, ...) {
  cat(as.character(x), "\n", sep = "")
  invisible(x)
}

coef.pyfcirt_model <- function(object, blocks = NULL, items = NULL, component = "measurement",
                              ci_prob = 0.95, ci_kind = "hdi", var_names = NULL, coords = list(),
                              kind = "all", digits = 3L, ...) {
  optional <- if (length(coords)) list(coords = coords) else list()
  table <- do.call(python_model(object)$coef,
                  c(list(blocks = selection(blocks), items = selection(items), component = component,
                         ci_prob = ci_prob, ci_kind = ci_kind, var_names = selection(var_names),
                         kind = kind), optional, list(...)))
  frame <- as_table(table)
  context <- to_r(table$attrs)
  for (field in setdiff(names(context), names(frame))) {
    frame[[field]] <- rep(context[[field]], nrow(frame))
  }
  structure(frame, class = c("pyfcirt_coef", "data.frame"), digits = digits)
}

print.pyfcirt_coef <- function(x, digits = attr(x, "digits"), ...) {
  table <- as.data.frame(x)
  for (name in intersect(c("mean", "sd", "median", "mcse_mean", "mcse_sd"), names(table))) {
    values <- table[[name]]
    table[[name]] <- ifelse(is.na(values), "", formatC(values, digits = digits, format = "f"))
  }
  interval <- grep("^(hdi|eti)", names(table), value = TRUE)
  for (name in interval) {
    values <- table[[name]]
    table[[name]] <- ifelse(is.na(values), "", formatC(values, digits = digits, format = "f"))
  }
  for (name in names(table)[vapply(table, is.list, logical(1L))]) {
    table[[name]] <- vapply(table[[name]], function(value) {
      paste(value[!is.na(value)], collapse = ", ")
    }, character(1L))
  }
  for (name in names(table)[vapply(table, is.numeric, logical(1L))]) {
    table[[name]][is.nan(table[[name]])] <- NA
  }
  print(table, row.names = FALSE, na.print = "", ...)
  invisible(x)
}

print.pyfcirt_traits <- function(x, digits = attr(x, "digits"), ...) {
  for (name in names(x)) {
    cat(name, "\n", sep = "")
    print(x[[name]], digits = digits, ...)
  }
  invisible(x)
}

response_counts <- function(answers, groupby = NULL) {
  frame <- reticulate::r_to_py(answers, convert = FALSE)
  # R list columns represent compound category and pair labels as vectors.
  columns <- list()
  for (name in intersect(c("pair", "category"), names(answers))) {
    if (is.list(answers[[name]])) {
      columns[[name]] <- lapply(answers[[name]], function(x) {
        if (length(x) > 1L) do.call(reticulate::tuple, as.list(x)) else x
      })
    }
  }
  if (length(columns)) frame <- do.call(frame$assign, columns)
  as_table(backend()$response_counts(frame, groupby = selection(groupby)))
}
