# 07_bayesian_robustness.R
#
# Supplementary Bayesian AR(1) robustness regression for the primary
# M2 model reported in Sections 2.5.1 and 3.2 of the manuscript.
#
# Two models are fit:
#   (1) Primary robustness:  mood ~ lag_mood + melatonin + agency + metacog
#   (2) Metacontrol:         mood ~ lag_mood + melatonin * metacog + agency
#
# Continuous variables are standardized (z-scored). Melatonin is 0/1.
#
# Priors (weakly informative):
#   coefficients b ~ Normal(0, 1)
#   intercept      ~ Student-t(3, 0, 2.5)
#   sigma          ~ Exponential(1)
#
# To run from the repo root:
#   setwd("/path/to/n-of-1-melatonin-study")
#   source("src/analysis/07_bayesian_robustness.R")

suppressPackageStartupMessages({
  library(brms)
  library(tidyverse)
  library(posterior)
})

# ---------- Load and prepare data ----------
df <- read.csv("outputs/clean_day.csv")

df_bayes <- df %>%
  arrange(study_day) %>%
  mutate(lag_mood = lag(mood)) %>%
  filter(!is.na(mood), !is.na(lag_mood),
         !is.na(agency), !is.na(metacognition), !is.na(melatonin)) %>%
  mutate(
    mood_z          = as.numeric(scale(mood)),
    lag_mood_z      = as.numeric(scale(lag_mood)),
    agency_z        = as.numeric(scale(agency)),
    metacognition_z = as.numeric(scale(metacognition)),
    melatonin       = as.integer(melatonin)
  )

cat(sprintf("Sample size: n = %d days (Day 18-70 subsample)\n", nrow(df_bayes)))

# ---------- Priors ----------
priors <- c(
  prior(normal(0, 1),         class = "b"),
  prior(student_t(3, 0, 2.5), class = "Intercept"),
  prior(exponential(1),       class = "sigma")
)

# ---------- Model 1: Primary Bayesian M2 robustness ----------
cat("\n===== Fitting Model 1: Bayesian M2 (primary robustness) =====\n")
fit_m2 <- brm(
  mood_z ~ lag_mood_z + melatonin + agency_z + metacognition_z,
  data    = df_bayes,
  family  = gaussian(),
  prior   = priors,
  chains  = 4,
  iter    = 4000,
  warmup  = 1000,
  cores   = 4,
  seed    = 20260511,
  control = list(adapt_delta = 0.95),
  refresh = 200
)

summary(fit_m2)

post_m2 <- as_draws_df(fit_m2)
summ_m2 <- summarise_draws(
  post_m2,
  mean, median, sd,
  ~quantile(.x, probs = c(0.025, 0.975)),
  rhat, ess_bulk, ess_tail,
  pr_gt_0 = ~mean(.x > 0)
)
print(summ_m2, n = Inf)

write.csv(summ_m2, "outputs/bayesian_m2_posterior_summary.csv", row.names = FALSE)
saveRDS(fit_m2, "outputs/bayesian_m2_fit.rds")

# ---------- Model 2: Metacontrol (secondary) ----------
cat("\n===== Fitting Model 2: Bayesian metacontrol (secondary) =====\n")
fit_meta <- brm(
  mood_z ~ lag_mood_z + melatonin * metacognition_z + agency_z,
  data    = df_bayes,
  family  = gaussian(),
  prior   = priors,
  chains  = 4,
  iter    = 4000,
  warmup  = 1000,
  cores   = 4,
  seed    = 20260511,
  control = list(adapt_delta = 0.95),
  refresh = 200
)

summary(fit_meta)

post_meta <- as_draws_df(fit_meta)
summ_meta <- summarise_draws(
  post_meta,
  mean, median, sd,
  ~quantile(.x, probs = c(0.025, 0.975)),
  rhat, ess_bulk, ess_tail,
  pr_gt_0 = ~mean(.x > 0)
)
print(summ_meta, n = Inf)

write.csv(summ_meta, "outputs/bayesian_metacontrol_posterior_summary.csv", row.names = FALSE)
saveRDS(fit_meta, "outputs/bayesian_metacontrol_fit.rds")

# ---------- Table-ready summaries (reported in the repository, cited from Section 3.4) ----------
fmt_ci <- function(lo, hi) sprintf("[%.2f, %.2f]", lo, hi)

# Primary Bayesian M2
m2_map <- c(b_Intercept = "Intercept", b_lag_mood_z = "mood_t-1 (z)",
            b_melatonin = "melatonin", b_agency_z = "agency (z)",
            b_metacognition_z = "metacognition (z)", sigma = "sigma")
tt <- summ_m2[match(names(m2_map), summ_m2$variable), ]
m2_table <- data.frame(
  Term        = unname(m2_map),
  Posterior_M = round(tt$mean, 2),
  CrI_95      = fmt_ci(tt$`2.5%`, tt$`97.5%`),
  Pr_gt_0     = round(tt$pr_gt_0, 3),
  Rhat        = round(tt$rhat, 3),
  ESS_bulk    = round(tt$ess_bulk)
)
write.csv(m2_table, "outputs/bayesian_m2_table.csv", row.names = FALSE)
cat("\n=== Bayesian M2 summary ===\n"); print(m2_table)

# Secondary: melatonin x metacognition interaction
int_var <- grep("^b_melatonin.*metacognition_z$", summ_meta$variable, value = TRUE)
if (length(int_var) == 1) {
  r <- summ_meta[summ_meta$variable == int_var, ]
  cat(sprintf("\nMetacontrol interaction: beta = %.2f, 95%% CrI %s, Pr(beta<0) = %.2f\n",
              r$mean, fmt_ci(r$`2.5%`, r$`97.5%`), 1 - r$pr_gt_0))
}

# Reproducibility log
writeLines(capture.output(sessionInfo()), "outputs/bayesian_sessionInfo.txt")

cat("\nDone. Output files written to outputs/:\n")
cat("  bayesian_m2_posterior_summary.csv\n")
cat("  bayesian_m2_table.csv              <- Bayesian M2 summary table\n")
cat("  bayesian_metacontrol_posterior_summary.csv\n")
cat("  bayesian_m2_fit.rds / bayesian_metacontrol_fit.rds\n")
cat("  bayesian_sessionInfo.txt\n")
