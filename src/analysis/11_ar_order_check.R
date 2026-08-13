# 11_ar_order_check.R
#
# Lag-order check across sampling RESOLUTIONS, reported in Appendix A. Fits
# AR(1) and AR(2) by maximum likelihood on the full daily series and on the
# timestamp-ordered observation series, comparing them by AIC/BIC and a
# likelihood-ratio test, with a Ljung-Box test on the AR(1) residuals. At the
# observation level a second lag is significant, consistent with within-day
# structure and the resolution-specific reading in Section 3.4.
#
# This script does NOT justify the lag order of the primary models. Those are
# estimated on the Day 18-70 window, so their lag-order sensitivity is checked
# on that same window in 02_ar1_models.py, using the same OLS machinery so the
# AIC/BIC values are comparable. See Section 2.5.1.
#
# Reads the cleaned analytic dataset, so the analysis-inclusion rule applied in
# 01_data_prep.py carries through here as well.
#
# Run from the repo root:
#   source("src/analysis/11_ar_order_check.R")

daily <- read.csv("outputs/clean_day.csv")
ema   <- read.csv("outputs/clean_obs.csv")
ema   <- ema[order(ema$datetime), ]

rows <- list()

check_order <- function(x, label, tag) {
  x <- x[!is.na(x)]
  cat(sprintf("\n== %s (n = %d) ==\n", label, length(x)))
  fits <- lapply(1:2, function(p) arima(x, order = c(p, 0, 0), method = "ML"))
  for (p in 1:2) {
    m   <- fits[[p]]
    co  <- coef(m)
    se  <- sqrt(diag(vcov(m)))
    ar  <- grep("^ar", names(co))
    est <- paste(sprintf("phi%d = %.3f [%.3f, %.3f]", seq_along(ar),
                         co[ar], co[ar] - 1.96 * se[ar], co[ar] + 1.96 * se[ar]),
                 collapse = "; ")
    cat(sprintf("AR(%d): AIC = %.2f, BIC = %.2f | %s\n", p, AIC(m), BIC(m), est))
  }
  lrt <- 2 * (logLik(fits[[2]]) - logLik(fits[[1]]))
  cat(sprintf("LRT AR(2) vs AR(1): chi2(1) = %.3f, p = %.3f\n",
              lrt, pchisq(lrt, 1, lower.tail = FALSE)))
  lb <- Box.test(residuals(fits[[1]]), lag = 10, type = "Ljung-Box", fitdf = 1)
  cat(sprintf("Ljung-Box on AR(1) residuals (lag 10): p = %.3f\n", lb$p.value))

  # Persist so the Appendix A numbers can be checked from outputs/.
  co2 <- coef(fits[[2]]); se2 <- sqrt(diag(vcov(fits[[2]])))
  rows[[tag]] <<- data.frame(
    series      = tag,
    n           = length(x),
    ar1_aic     = AIC(fits[[1]]), ar1_bic = BIC(fits[[1]]),
    ar2_aic     = AIC(fits[[2]]), ar2_bic = BIC(fits[[2]]),
    phi1_ar1    = coef(fits[[1]])[["ar1"]],
    phi2_ar2    = co2[["ar2"]],
    phi2_ci_lo  = co2[["ar2"]] - 1.96 * se2[["ar2"]],
    phi2_ci_hi  = co2[["ar2"]] + 1.96 * se2[["ar2"]],
    lrt_chi2    = as.numeric(lrt), lrt_p = pchisq(lrt, 1, lower.tail = FALSE),
    ljungbox_p  = lb$p.value,
    stringsAsFactors = FALSE
  )
}

check_order(daily$mood, "daily mean mood, full 70-day series", "daily_full70")
check_order(ema$mood,
            sprintf("observation-level mood, %d scheduled pings", nrow(ema)),
            "observation_level")

write.csv(do.call(rbind, rows), "outputs/ar_order_by_resolution.csv", row.names = FALSE)
cat("\nWrote outputs/ar_order_by_resolution.csv\n")
