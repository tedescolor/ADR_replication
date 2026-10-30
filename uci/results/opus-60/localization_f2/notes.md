# Notes

## Data
85 rows, y > 0 (0.006 .. 1.09), right-skewed; log y is much closer to symmetric. Covariates look like a
designed experiment: x0 ~ {10,15,20,..,30}, x1 ~ {12,15,..,25}, x2 at exactly 3 levels (~100, 200, 300),
x3 ~ 15..100, while x4 is continuous. Strongest signals: x4 (+, log-log slope ~1, i.e. y ∝ x4) and x2 (-).
One gross outlier in log scale: row 64 (y = 0.0058, ~-6 residual sd under a lognormal fit).

## Validation
Leave-one-out CV (refit by maximum likelihood for every left-out row), mean NLL on the original y scale.
Because row 64 dominates LOO means (one point can contribute 18 nats / 85 = 0.2), I also tracked LOO
excluding that row and the median.

## What was tried (LOO mean NLL)
- Lognormal, log-linear mean in all 5 log covariates: -0.97 (row 64 costs ~18 nats).
- Log-t (fitted df): -0.94..-1.07; when row 64 is left out the df fit goes to ~normal, so it doesn't help.
- Gamma (log link): -1.00..-1.12 (best with x2 as factor + log x1 + log x4).
- Lognormal mixture with fitted contamination: -0.91..-1.09 (same collapse as log-t).
- Heteroscedastic scale (on log x2, log x4, x2 levels): consistently worse.
- Extra terms (log x0, log x3, x2*x4 interaction, (log x4)^2): no LOO gain.
- **Lognormal + fixed contamination** (weight W, same median, scale K times larger):
  mean spec {x2 factor, log x1, log x4}: -1.227 at W=0.03, K=4. Surface is flat
  (W 0.01..0.05, K 3..8 all give -1.221..-1.227); gamma-based variant: -1.20.
  With x2 as continuous log x2 instead of factor: -1.200.

## Final program
log y median = 1.8815 - 0.386*[x2>150] - 0.5447*[x2>250] - 1.0674*log x1 + 0.9931*log x4;
y ~ 0.97 LogNormal(mu, 0.2954) + 0.03 LogNormal(mu, 1.1816). 10 numeric constants.
Picked because it had the best LOO, few parameters, and the fixed heavy component guards against
rare outliers like row 64 without being fitted away in CV. Training NLL -1.305, LOO -1.227.

## Expected NLL
LOO is -1.227; allowing for a little selection optimism (hyperparameters W, K and the
feature set were chosen on the same LOO) and uncertainty in how often outliers turn up, I expect about -1.20.

expected_nll: -1.20
