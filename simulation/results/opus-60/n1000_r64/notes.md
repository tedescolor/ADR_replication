# Notes

## Data
1000 rows, x0 in (0,1). Scatter shows one noisy curve for x0 < ~0.5 (dip to about -0.9 near
x0=0.25) that splits into two branches for x0 > 0.5: one rising nearly linearly to about 2.5 at x0=1,
the other a hump falling to about -1 at x0=1. The noise sd shrinks with x0 (from ~0.3 at 0 to ~0.09 at 1).

## Candidates (Gaussian 2-component mixtures, fit by maximum likelihood with BFGS)
Basis t = 2*x0-1. Validation used 5-fold CV repeated twice (10 held-out folds), with the full-data
fit as the warm start for each fold. Mean held-out NLL:
- hard split point c, B mean + hinge*poly for A (first attempt): 0.20-0.23 (optimisation was unstable)
- free means of degree 5/6/7, separate quadratic log-sds, constant weight: 0.196 / 0.192 / 0.195
- deg 6, separate log-sd plus a logistic-polynomial weight: 0.207 (overfits)
- deg 6, shared log-sd, linear / quadratic: 0.195 / 0.191
- deg 6, separate quadratic log-sd: 0.194
- **deg 6, shared sd linear in x, weight fixed at 0.5: 0.189** (training NLL 0.171)
- the same with a free weight: 0.191 (fitted weight about 0.5)
- deg 5 / deg 7 / quadratic sd with linear sd: 0.191 / 0.191 / 0.192

The differences are small (within about 0.005), so I picked the simplest model with the best CV score.
For x0 < 0.5 the two mean curves nearly coincide (the mixture behaves like one curve, slightly
non-Gaussian). For x0 > 0.5 they follow the two branches with equal weight.

## Final program
Mixture(0.5/0.5) of Normal(ma(t), s(t)) and Normal(mb(t), s(t)): ma and mb are degree-6 polynomials,
s = max(a + b t, 0.001) is shared by both components. The fitted parameters are 16 numbers.

Estimate: CV is about 0.189 and training NLL is 0.171. I expect around 0.19 on new data.

expected_nll: 0.19
