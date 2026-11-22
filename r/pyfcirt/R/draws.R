draw_analysis <- function() reticulate::import("pyfcirt.analysis.summary", convert = FALSE)

axis_labels <- function(value, name) {
  coordinate <- reticulate::py_get_item(value$coords, name)
  to_r(draw_analysis()$`_draw_labels`(coordinate)$tolist())
}

select_labels <- function(value, coords) {
  if (length(coords)) value$sel(lapply(coords, as.list)) else value
}

draw_component <- function(value, name) {
  dims <- unlist(to_r(value$dims), use.names = FALSE)
  sample_dims <- c("draw", "chain")
  other <- setdiff(dims, sample_dims)
  value <- do.call(value$transpose, as.list(c(sample_dims, other)))
  values <- to_r(value$values)
  n <- dim(values)
  indices <- if (length(other)) {
    expand.grid(stats::setNames(lapply(n[-c(1L, 2L)], seq_len), other), KEEP.OUT.ATTRS = FALSE)
  } else {
    data.frame(row.names = 1L)
  }

  labels <- list()
  coords <- list()
  draw_labels <- draw_analysis()$`_draw_labels`
  for (key in setdiff(python_names(value$coords$keys()), sample_dims)) {
    coordinate <- reticulate::py_get_item(value$coords, key)
    axes <- unlist(to_r(coordinate$dims), use.names = FALSE)
    if (length(axes) && all(axes %in% other)) {
      raw <- to_r(draw_labels(coordinate))
      if (key %in% other) {
        labels[[key]] <- to_r(draw_labels(coordinate, format_names = TRUE))[indices[[key]]]
      }
      coords[[key]] <- if (length(axes) == 1L) {
        as.vector(raw[indices[[axes]]])
      } else {
        raw[as.matrix(indices[axes])]
      }
    }
  }
  names <- name
  if (length(other)) {
    names <- paste0(name, "[", do.call(paste, c(unname(labels[other]), sep = ",")), "]")
  }
  keys <- data.frame(variable = names, parameter = name)
  array <- array(values, dim = c(n[1:2], nrow(indices)), dimnames = list(NULL, NULL, names))
  list(array = array, keys = keys, coords = coords)
}

as_draws <- function(m, variables = NULL, coords = list(), chain = NULL, draw = NULL) {
  if (length(coords) && (is.null(names(coords)) || any(!nzchar(names(coords))) ||
                        anyDuplicated(names(coords)) ||
                        any(names(coords) %in% c("chain", "draw")))) {
    stop("coords must have unique names. Use chain and draw for sample labels.", call. = FALSE)
  }
  labels <- Filter(Negate(is.null), c(list(chain = chain, draw = draw), coords))
  selected <- draw_analysis()$`_select_posterior_draws`(
    python_model(m), selection(unique(variables)),
    if (length(labels)) lapply(labels, as.list) else reticulate::dict()
  )
  variables <- python_names(selected$data_vars$keys())
  parts <- lapply(variables, function(name) {
    draw_component(reticulate::py_get_item(selected, name), name)
  })
  dims <- dim(parts[[1L]]$array)[1:2]
  keys <- do.call(rbind, lapply(parts, "[[", "keys"))
  values <- unlist(lapply(parts, "[[", "array"), use.names = FALSE)
  draws <- posterior::as_draws_array(array(values, c(dims, nrow(keys)),
                                          dimnames = list(NULL, NULL, keys$variable)))
  attr(draws, "coordinates") <- stats::setNames(lapply(parts, function(part) {
    c(list(variable = part$keys$variable), part$coords)
  }), variables)
  attr(draws, "sample_coordinates") <- list(chain = axis_labels(selected, "chain"),
                                           draw = axis_labels(selected, "draw"))
  draws
}
