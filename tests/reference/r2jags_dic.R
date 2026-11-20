# Execute the original pinned R2jags conversion functions, not a replacement DIC formula.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3L)
reference_source <- args[[3]]
reference_library <- Sys.getenv("PYFCIRT_R_LIBRARY")
if (nzchar(reference_library)) .libPaths(c(reference_library, .libPaths()))
stopifnot(requireNamespace("R2WinBUGS", quietly = TRUE))
description <- read.dcf(file.path(reference_source, "DESCRIPTION"))
stopifnot(description[1, "Package"] == "R2jags", description[1, "Version"] == "0.9-0")
# The post-processing functions need monitor and the parameter-name decoder;
# the JAGS sampling engine is not involved. No source files are modified.
reference <- new.env(parent = asNamespace("R2WinBUGS"))
for (file in c("jags.sims.R", "as.bugs.array2.R", "mcmc2bugs.R")) {
    sys.source(file.path(reference_source, "R", file), envir = reference)
}
deviance <- as.matrix(read.csv(args[[1]], check.names = FALSE))
convert <- function(values) {
    draws <- nrow(values)
    chains <- ncol(values)
    simulations <- array(0, dim = c(draws, chains, 2L),
                         dimnames = list(NULL, NULL, c("anchor", "deviance")))
    simulations[, , "anchor"] <- seq_len(draws) / draws
    simulations[, , "deviance"] <- values
    reference$as.bugs.array2(simulations, program = "jags", DIC = TRUE,
                            DICOutput = NULL, n.burnin = 0)
}
set.seed(246)
all_chains <- convert(deviance)
rows <- lapply(seq_len(ncol(deviance)), function(chain) {
    converted <- convert(deviance[, chain, drop = FALSE])
    data.frame(chain = colnames(deviance)[chain], Dbar = mean(deviance[, chain]),
               pV = converted$pV, DIC = converted$DIC)
})
rows[[length(rows) + 1L]] <- data.frame(chain = "all", Dbar = mean(deviance),
                                       pV = all_chains$pV, DIC = all_chains$DIC)
write.csv(do.call(rbind, rows), args[[2]], row.names = FALSE)
metadata <- data.frame(
    R = as.character(getRversion()), R2jags_source = description[1, "Version"],
    R2WinBUGS = as.character(utils::packageVersion("R2WinBUGS")),
    mode = "unmodified-pinned-source-postprocessing-no-JAGS"
)
write.csv(metadata, paste0(args[[2]], ".metadata.csv"), row.names = FALSE)
