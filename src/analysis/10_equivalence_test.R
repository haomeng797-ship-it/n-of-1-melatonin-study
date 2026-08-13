# 10_equivalence_test.R
#
# Two one-sided tests (TOST) equivalence test for the melatonin mean effect on
# daily mood, reported in Sections 2.5.10 and 3.3 of the manuscript. Complements
# the between-condition Welch test by asking whether the effect is small enough
# to be declared equivalent to zero within a pre-specified bound (SESOI).
# Reports SESOI = +/-5 points (primary; the smallest practically meaningful
# day-level shift, ~0.9 of the daily mood SD) and a tighter +/-3 points.
# Uses Lakens's TOSTER package; the effect is melatonin minus control.
#
# Run from the repo root, after the Python pipeline has written outputs/:
#   setwd("/path/to/N-of-1-Melatonin-Study")
#   source("src/analysis/10_equivalence_test.R")

suppressPackageStartupMessages({
  library(TOSTER)
})

day  <- read.csv("outputs/clean_day.csv")
ctrl <- day$mood[day$condition == "control"   & !is.na(day$mood)]
mel  <- day$mood[day$condition == "melatonin" & !is.na(day$mood)]

# x = melatonin, y = control  ->  estimate is mel - ctrl (matches the manuscript)
run_tost <- function(delta) {
  TOSTER::t_TOST(x = mel, y = ctrl, eqb = delta, eqbound_type = "raw",
                 var.equal = FALSE, hypothesis = "EQU", alpha = 0.05)
}
res5 <- run_tost(5)
res3 <- run_tost(3)

cat("=== SESOI +/-5 points ===\n"); print(res5)
cat("\n=== SESOI +/-3 points ===\n"); print(res3)

# ---------- shared descriptives ----------
n1 <- length(ctrl); n2 <- length(mel)
diff <- mean(mel) - mean(ctrl)
pooled_sd <- sqrt(((n1 - 1) * var(ctrl) + (n2 - 1) * var(mel)) / (n1 + n2 - 2))
# Welch t-test row is identical across res5/res3; 90% CI = the interval TOST checks
welch <- res5$TOST["t-test", ]
ci90  <- as.numeric(res5$effsize["Raw", c("lower.ci", "upper.ci")])

# ---------- per-SESOI rows ----------
mk <- function(res, delta) {
  p_tost  <- max(res$TOST["TOST Lower", "p.value"], res$TOST["TOST Upper", "p.value"])
  d_bound <- delta / pooled_sd
  tag <- sub("\\.", "", as.character(delta))
  data.frame(
    key = c(sprintf("tost.sesoi%s.delta", tag),
            sprintf("tost.sesoi%s.d_bound", tag),
            sprintf("tost.sesoi%s.p", tag),
            sprintf("tost.sesoi%s.equivalent", tag)),
    value = c(delta, round(d_bound, 3), signif(p_tost, 4), as.integer(p_tost < 0.05)),
    description = c(sprintf("SESOI bound in points, +/-%g", delta),
                    sprintf("SESOI as Cohen's d, +/-%g", delta),
                    sprintf("TOST p at +/-%g (max of two one-sided tests)", delta),
                    sprintf("equivalence established at +/-%g (1/0)", delta)),
    stringsAsFactors = FALSE)
}

desc <- data.frame(
  key = c("desc.control_n", "desc.melatonin_n", "desc.diff", "desc.welch_t",
          "desc.welch_df", "desc.welch_p", "desc.cohens_d",
          "tost.ci90_lo", "tost.ci90_hi"),
  value = c(n1, n2, round(diff, 4),
            round(welch[["t"]], 4), round(welch[["df"]], 4), round(welch[["p.value"]], 4),
            round(diff / pooled_sd, 4), round(ci90[1], 4), round(ci90[2], 4)),
  description = c("n control days", "n melatonin days",
                  "mean difference (melatonin - control), 0-100 mood",
                  "Welch t for mean difference", "Welch df",
                  "two-sided p for mean difference", "Cohen's d (pooled)",
                  "lower 90% CI of mean difference", "upper 90% CI of mean difference"),
  stringsAsFactors = FALSE)

out <- rbind(desc, mk(res5, 5), mk(res3, 3))
write.csv(out, "outputs/equivalence_tost.csv", row.names = FALSE)

# ---------- TOSTER equivalence plot (supplementary; not a manuscript figure) ----------
png("figures/equivalence_tost_plot.png", width = 1500, height = 950, res = 160)
plot(res5)
dev.off()

writeLines(capture.output(sessionInfo()), "outputs/equivalence_sessionInfo.txt")

cat("\n=== equivalence_tost.csv ===\n"); print(out, row.names = FALSE)
cat("\nWrote outputs/equivalence_tost.csv, figures/equivalence_tost_plot.png\n")
