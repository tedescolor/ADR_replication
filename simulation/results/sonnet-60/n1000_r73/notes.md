# Notes

Data: 1 covariate x0 in [0,1], y with a nonlinear mean and a spread that grows with x. For x > 0.8, y is bimodal: one branch near -0.5 to -1 and one near 2.3.

Tried (5-fold CV, mean NLL, same folds for all candidates, seed 0):
- Single Normal / StudentT / SkewNormal with polynomial mean and log-sd: best about 0.617 (Normal, quartic mean, quadratic log-sd). The t and skew-normal were no better.
- Two-component Normal mixture with polynomial component means, polynomial log-sds and a polynomial logit weight:
  - (4,1,1,1,1) 0.2225
  - (4,1,2,1,2) 0.2117
  - (4,1,2,1,3) 0.207
  - (4,2,2,1,2) 0.1953 <- chosen
  - Orders are the polynomial degrees for (mean A, mean B, log-sd A, log-sd B, logit weight).

Final program: Mixture of 2 Normals. The logit weight is quadratic in u = x0 - 0.5. Mean A is quartic and mean B is quadratic. Log-sd A is quadratic and log-sd B is linear. There are 16 fitted constants, plus a few structural ones (0.5, 1), well under the limit of 80.
Fitted on the full data with the best of 3 random starts (train NLL 0.175).

Validation caveat: only a single 5-fold CV run per candidate, so differences of about 0.01 between candidates are noise. The structure was picked from about 15 candidates, which adds slight optimism. I add about 0.01 to the CV figure.

expected_nll: 0.205
