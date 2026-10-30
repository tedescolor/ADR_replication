# Notes

## Data
1000 rows, x0 in (0,1). Looking at the scatter: for x < ~0.5 there is one noisy curve (sd ~0.3, dips to -0.8 near x=0.25).
Above x ~0.55 it splits into two clean branches. The upper branch rises to ~2.5 at x=1 and the lower one falls to ~-1 at x=1.
Noise shrinks as x grows (sd ~0.1 at x=1). The two branches hold about equal shares of the points.

## Model family
I used a 2-component Normal mixture, fitted by maximum likelihood (L-BFGS then BFGS). With t = 2x-1, each
component mean is a polynomial in t, each log-sd is a polynomial in t, and the mixture logit is a polynomial in t.
Degrees are written as (mean1, logsd1, mean2, logsd2, logit).
Higher-degree fits are warm-started from the (6,2,3,1,1) fit on the same training fold.

## Validation
I used K-fold CV and refitted everything inside each fold. Out-of-sample mean NLL:
- (5,2,2,1,1): 0.213 (10-fold)
- (6,2,3,1,1): 0.178 (10-fold), 0.176 (5-fold)
- (6,2,3,1,0): 5-fold CV of 0.175, 0.183 and 0.174 for seeds 0, 1 and 2, mean ≈ 0.178 (selected)
- (6,1,3,1,0): 0.176. (7,2,3,1,0): 0.176, 0.186, 0.175. (5,2,3,1,0): 0.186, 0.180.
- (6,2,2,1,0): 0.26. (7,2,4,1,0): 0.179. (8,2,5,2,0): 0.184. (6,3,4,2,0): 0.181.
- Student-t components with a shared df: the fitted df was ~40 and gave no gain (0.1753), so Normal it is.

The weight came out constant: the fitted value is about 0.48/0.52, and the x-dependent logit did not help.

## Final
I chose (6,2,3,1,0) because it had the best CV, averaged over seeds, among the candidates. It has 20 literal constants:
17 fitted ones, plus the 2 and 1 in t=2x-1 and the 1 in 1-w. Training NLL is 0.157 and CV is about 0.178. I add a small allowance for refitting on all the data.

expected_nll: 0.178
