# Later-stage Cox model with a gamma frailty term per project (called by rq1_extra.py).
# Usage: Rscript frailty_cox.R input.csv output.csv
suppressMessages(library(survival))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[1])
feats <- setdiff(names(d), c("duration", "event", "project_url"))
d$project_url <- factor(d$project_url)
rhs <- paste(feats, collapse = " + ")

tidy <- function(fit, spec, var = NA) {
  s <- summary(fit)$coefficients
  # penalized fits truncate row names: the covariates come first, in formula order
  s <- s[seq_along(feats), , drop = FALSE]
  rownames(s) <- feats
  se <- if ("se(coef)" %in% colnames(s)) s[, "se(coef)"] else s[, "se2"]
  data.frame(spec = spec, covariate = rownames(s), HR = exp(s[, "coef"]),
             ci_low = exp(s[, "coef"] - 1.96 * se), ci_high = exp(s[, "coef"] + 1.96 * se),
             p = if ("p" %in% colnames(s)) s[, "p"] else s[, "Pr(>|z|)"],
             frailty_variance = var, row.names = NULL)
}

f0 <- coxph(as.formula(paste("Surv(duration, event) ~", rhs)), data = d, ties = "efron")
f1 <- coxph(as.formula(paste("Surv(duration, event) ~", rhs, "+ frailty(project_url, distribution = 'gamma')")),
            data = d, ties = "efron", control = coxph.control(iter.max = 200, outer.max = 50))
out <- rbind(tidy(f0, "no_frailty_unpenalized"), tidy(f1, "gamma_frailty", f1$history[[1]]$theta))
write.csv(out, args[2], row.names = FALSE)
