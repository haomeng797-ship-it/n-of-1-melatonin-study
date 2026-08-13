# 09_multilevel_robustness.R
#
# Multilevel (mixed-effects) robustness model reported in Sections 2.5.10 and
# 3.4 of the manuscript (Table B.1). Reproduces, from outputs/clean_obs.csv:
#
#   mood_ij = g00 + g10*agency_ij + g20*metacognition_ij + g30*melatonin_j
#             + u0j + e_ij,        u0j ~ N(0, tau00),  e_ij ~ N(0, sigma^2)
#
# Momentary EMA observations (i) nested within study day (j); agency and
# metacognition grand-mean-centered; melatonin a day-level 0/1 indicator;
# random intercept for day. Restricted to the Day 18-70 window. ICC is taken
# from unconditional models on the full analytic sample and on the
# Day 18-70 window. Marginal/conditional R^2 follow Nakagawa & Schielzeth
# (computed directly from the variance components, no extra packages).
#
# Run from the repo root, after the Python pipeline has written outputs/:
#   setwd("/path/to/N-of-1-Melatonin-Study")
#   source("src/analysis/09_multilevel_robustness.R")

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)   # Satterthwaite df / p-values for fixed effects
})

obs <- read.csv("outputs/clean_obs.csv")

# ---------- Unconditional ICC (full sample and Day 18-70 window) ----------
icc_uncond <- function(df) {
  m <- lmer(mood ~ 1 + (1 | study_day), data = df, REML = TRUE)
  v <- as.data.frame(VarCorr(m))
  tau <- v$vcov[v$grp == "study_day"]
  sig <- v$vcov[v$grp == "Residual"]
  tau / (tau + sig)
}
full <- subset(obs, !is.na(mood))
win  <- subset(obs, study_day >= 18 &
                    !is.na(mood) & !is.na(agency) & !is.na(metacognition))
icc_full <- icc_uncond(full)
icc_win  <- icc_uncond(win)

# ---------- Grand-mean-center predictors (Day 18-70 momentary means) ----------
ag_m <- mean(win$agency)
mc_m <- mean(win$metacognition)
win$agency_c <- win$agency - ag_m
win$meta_c   <- win$metacognition - mc_m

# ---------- Conditional model (Table B.1) ----------
m <- lmer(mood ~ agency_c + meta_c + melatonin + (1 | study_day),
          data = win, REML = TRUE)
fe <- summary(m)$coefficients                       # Estimate, SE, df, t, Pr(>|t|)
ci <- confint(m, parm = "beta_", method = "Wald")   # Wald CIs for fixed effects

# Variance components + Nakagawa R^2 (random-intercept model)
vc  <- as.data.frame(VarCorr(m))
tau <- vc$vcov[vc$grp == "study_day"]
sig <- vc$vcov[vc$grp == "Residual"]
Xb   <- predict(m, re.form = NA)                    # fixed-effects-only prediction
varF <- var(Xb)
R2m  <- varF / (varF + tau + sig)                   # marginal
R2c  <- (varF + tau) / (varF + tau + sig)           # conditional

# ---------- Table B.1 (paste straight in) ----------
rows <- c("(Intercept)", "agency_c", "meta_c", "melatonin")
labs <- c("Intercept", "Agency (centered)", "Metacognition (centered)", "Melatonin")
pfmt <- function(p) ifelse(p < .001, "< .001", sub("^0", "", sprintf("%.3f", p)))
tab <- data.frame(
  Term = labs,
  beta = round(fe[rows, "Estimate"], 2),
  SE   = round(fe[rows, "Std. Error"], 2),
  CI95 = sprintf("[%.2f, %.2f]", ci[rows, 1], ci[rows, 2]),
  p    = pfmt(fe[rows, "Pr(>|t|)"]),
  row.names = NULL
)
write.csv(tab, "outputs/multilevel_table.csv", row.names = FALSE)

stats <- data.frame(
  key = c("n_obs", "n_groups", "n_control_days", "n_melatonin_days",
          "icc_full_sample", "icc_window_day18_70",
          "random_intercept_var", "residual_var",
          "marginal_R2", "conditional_R2", "center_agency", "center_metacog"),
  value = c(nrow(win), length(unique(win$study_day)),
            length(unique(win$study_day[win$melatonin == 0])),
            length(unique(win$study_day[win$melatonin == 1])),
            round(icc_full, 3), round(icc_win, 3),
            round(tau, 2), round(sig, 2),
            round(R2m, 2), round(R2c, 2),
            round(ag_m, 2), round(mc_m, 2))
)
write.csv(stats, "outputs/multilevel_fit_stats.csv", row.names = FALSE)
writeLines(capture.output(sessionInfo()), "outputs/multilevel_sessionInfo.txt")

cat("\n=== Table B.1 (paste into manuscript) ===\n"); print(tab)
cat("\n=== Fit statistics (Table B.1 note) ===\n"); print(stats)
cat("\nWrote outputs/multilevel_table.csv, outputs/multilevel_fit_stats.csv\n")
