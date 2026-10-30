# Notes

## Data findings
- OLS on all 25 covariates: residual sd 0.44 with strongly left-skewed residuals. A linear mean plus Student-t noise gives CV NLL ≈ 0.596.
- A 2-normal mixture fitted on single features showed that y ≈ x24 − 0.04 (sd 0.027) for about 20% of rows.
  Also y ≤ x24 + 0.02 in every row. So I modelled D = x24 − y ≥ 0.
- x24 is about max(x1..x23), and sometimes larger. Take G = x24 − max(x1..x23). G is about 0 ± 0.02 for most rows.
  When G > ~0.015, the probability that D sits in the narrow spike (~0.034, sd 0.022) goes from ~0.05 to ~0.6,
  and the D tail moves a little lower. Apart from x24 and G, the covariates barely predict D (|corr| < 0.15).

## Validation
5-fold CV repeated 3 times (15 refits per model, MLE by L-BFGS). The CV fold fits start from the full-data
optimum, which leaks a little into initialization only. Mean test NLL:
- linear mean + t: 0.596; linear mean + 2-normal mixture: ~0.49
- y = x24 − D, D ~ normal mixture: K=2 0.469, K=3 0.392, K=4 0.352, K=5 0.349
- + soft step h = sigmoid((G − g0)/t) on the spike weight and a tail shift: K=3 0.209, K=4 0.175, K=5 0.178
- K=4 + tail weights depending on h: **0.169** (chosen)
- extra x0 effects 0.176; tail scale depending on h 0.184 (neither helped)

## Final program
Mixture of 4 normals in y. Spike component at x24 − 0.034. Three tail components at x24 + 0.051h − mu_k.
The weights are log-linear in h, where h is a steep sigmoid of x24 − max(x1..x23) − 0.015. 19 constants.
Training NLL 0.148; CV 0.169. I allow a little for selection and init optimism.

expected_nll: 0.18
