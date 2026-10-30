# Notes

## Data
250 rows, x0 in [0,1]. For x0 < ~0.5, y follows a single wavy curve (min ~-0.87 at x~0.25,
sd ~0.3). From x0 ~0.5 the data split into two branches: an upper branch rising linearly
(~1.0 at 0.65 to ~2.6 at 1.0, sd ~0.14) and the main curve continuing down to ~-0.9 at x=1,
with noise shrinking to ~0.1. About 45% of points at high x are on the upper branch.
Residuals within branches look Gaussian (skew/kurtosis ~0).

## Candidates (all ML fits, 2x repeated 10-fold CV NLL, same folds)
2-component Gaussian mixture, gate w(x) on the upper component:
- main curve as degree-4..6 polynomial, constant sds: CV 0.30-0.32
- main curve a0+a1 x+a2 sin(a3 x+a4), constant sds: CV 0.297
- sin main curve, log-linear sds, plain sigmoid gate: CV 0.242-0.251
- poly-5 main curve, log-linear sds: CV 0.262
- sin, gate = sigmoid(k(x-c))*cap, log-sd1 linear, sd2 const: CV 0.243
- **sin, capped gate, log-sd1 quadratic, sd2 const: CV 0.238 (train 0.179)** <- chosen
- same with quadratic upper branch: CV 0.248 (worse)
Differences among the top few are ~0.01, within CV noise; chosen = best CV with 14 params.

## Final program
Mixture of Normal(m1, s1) and Normal(m2, s2):
m1 = 0.461 - 1.654x + 0.950 sin(5.358x + 3.845), log s1 quadratic in x,
m2 = -2.135 + 4.735x, s2 = exp(-1.968), w = sigmoid(9.6(x-0.506)) * sigmoid(-0.135).
Refit on all data; in-sample NLL 0.179. 14 fitted constants (+1 literal 1.0).

## Expected NLL
CV estimate 0.238; slight optimism from model selection vs slight pessimism from
training on 90% of data -> estimate ~0.24.

expected_nll: 0.24
