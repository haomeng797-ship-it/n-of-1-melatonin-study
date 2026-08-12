# 11_ar_order_check.R
#
# Lag-order check for the autoregressive specification, reported in Section
# 2.5.1 and Appendix A of the manuscript. Fits AR(1) and AR(2) by maximum
# likelihood at both sampling resolutions and compares them by AIC/BIC and a
# likelihood-ratio test, with a Ljung-Box test on the AR(1) residuals. At the
# daily level a second lag adds nothing, supporting the first-order primary
# models; at the observation level a second lag is significant, consistent
# with within-day structure and the resolution-specific reading in Section 3.4.
#
# Run from the repo root:
#   source("src/analysis/11_ar_order_check.R")

daily <- read.csv("data/miura_ema_70day_daily.csv")
ema   <- read.csv("data/miura_ema_70day.csv")
ema   <- ema[order(ema$datetime), ]

check_order <- function(x, label) {
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
}

check_order(daily$mood, "daily mean mood, full 70-day series")
check_order(ema$mood,   "observation-level mood, 195 pings")
