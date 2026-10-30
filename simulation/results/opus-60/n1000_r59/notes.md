# Notes

## Data
There is one covariate x0 in [0, 1] and the target y. The scatter plot shows two noisy curves.
For x > 0.55 they separate clearly: an upper branch rises to about 2.5 and a lower branch falls to about -1.
For x < 0.45 they overlap. There the residuals around one smooth mean are platykurtic (excess kurtosis about -0.5),
which fits two overlapping curves better than one. The noise scale falls from about 0.25 at x=0 to about 0.1 at x=1.
About half of the points are on each branch.

## Models tried
All candidates were 2-component Normal mixtures with polynomial means, a polynomial log-sd and a weight of
0.5, a constant or logistic in x. They were fitted by maximum likelihood (BFGS, several starts).
- Baseline: quartic means, separate quadratic log-sd per component, logistic weight (18 params). Train 0.160.
- Grid over mean degrees 3–5, log-sd degree 1–3, shared vs separate sd, weight half/const/logistic.

## Validation
Repeated 5-fold CV (2 repeats, 10 folds, same splits for every model). Each fold was refit from scratch
(plus a warm start from the full-data fit to avoid label-switching local optima). Selected CV NLLs:
- (4,4) means, separate sd, logistic w (baseline-like): 0.1868
- (4,4) means, shared quad log-sd, w=0.5: 0.1780
- (3,5) shared quad, w=0.5: 0.1771
- (3,4) shared quad, w=0.5: **0.1760** (train 0.1624)
- (3,3): 0.183; quad→cubic log-sd: 0.179; linear log-sd: about 0.180
The fold-level SE is about 0.016, but the comparisons are paired, so the more parsimonious structure
(shared sd, fixed equal weights) is favoured consistently.

## Final program
`Mixture([0.5,0.5], [Normal(cubic(x), s), Normal(quartic(x), s)])` with `s = exp(quadratic(x))`.
It uses 12 fitted constants (14 literals including the two 0.5 weights), refitted on all 1000 rows.

The CV estimate comes from fits on 800 rows. The full-data fit should be very slightly better, so my estimate is:

expected_nll: 0.175
