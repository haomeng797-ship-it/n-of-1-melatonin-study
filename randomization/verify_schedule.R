# verify_schedule.R
#
# Checks that the archived randomization schedule satisfies the constraints the
# protocol states: 35 melatonin and 35 control days over 70 days, with no run
# longer than two consecutive days in either condition.
#
# The schedule was generated before data collection by a constrained
# randomization procedure written for this study. The random seed was not
# recorded, so the exact draw cannot be regenerated; what this script does is
# verify that the archived schedule is a valid member of the constrained set.
# The same procedure was later incorporated into the R package nof1kit, whose
# design_schedule() draws uniformly from that set and whose check_schedule()
# reports the diagnostics used below.
#
# Run from the repo root:
#   source("randomization/verify_schedule.R")

suppressPackageStartupMessages({
  library(jsonlite)
})

sched <- fromJSON("randomization/schedule.json")
day   <- as.integer(names(sched))
cond  <- as.integer(unlist(sched))
ord   <- order(day)
day   <- day[ord]
cond  <- cond[ord]

n_days     <- length(cond)
n_active   <- sum(cond == 1L)
n_control  <- sum(cond == 0L)
runs       <- rle(cond)
max_run    <- max(runs$lengths)

cat(sprintf("days: %d\n", n_days))
cat(sprintf("melatonin: %d | control: %d\n", n_active, n_control))
cat(sprintf("longest run: %d\n", max_run))
cat("run lengths:\n"); print(table(runs$lengths))

stopifnot(
  n_days    == 70L,
  n_active  == 35L,
  n_control == 35L,
  max_run   <= 2L
)
cat("\nAll protocol constraints satisfied.\n")

# Optional: the same diagnostics via nof1kit, if it is installed.
if (requireNamespace("nof1kit", quietly = TRUE)) {
  df <- data.frame(day = day, condition = cond)
  class(df) <- c("nof1_schedule", "data.frame")
  cat("\nnof1kit::check_schedule():\n")
  print(nof1kit::check_schedule(df))
}
