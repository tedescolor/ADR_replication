# Conditional density estimation: notes

## Data
1000 rows, x0 in (0,1). Scatter (work/scatter.png): for x < ~0.55 y is one smooth curve
(dips to about -0.85 at x=0.25 and rises after that) with Gaussian noise whose sd falls from
about 0.37 to 0.2. Above about 0.55 the data split into two clean branches of roughly equal
weight: the upper branch rises to 2.5 at x=1, the lower one falls to -0.9. Their noise sd
falls from about 0.2 to 0.1. Standardized residuals look Gaussian (left part: skew 0.02,
excess kurtosis 0.07, Shapiro p=0.84), so heavy-tailed families were not needed (a Student-t
on the left gave no CV gain).

## Candidates (10-fold CV, mean held-out NLL)
1. A global 2-component normal mixture with polynomial means and log-polynomial sds, plus a
   sigmoid gate on the weight (fit2.py): CV about 0.199–0.21 depending on the degrees. The
   gate collapsed to a step at x≈0.547.
2. A piecewise model (fit3.py): for x<c, one Normal (polynomial mean, log-linear/quadratic sd);
   for x>=c, a 2-component mixture of polynomial branches with a constant weight. I searched a
   grid over c ∈ {0.5,...,0.6}, the degrees, and shared vs separate branch sds. The best
   configurations scored 0.1955–0.197, and many came within noise of each other.
   I re-ran the top 4 with 3 more CV seeds:
   - c=0.55, left quartic mean with log-linear sd, right quadratic branches with a shared log-linear sd: **0.1946**
   - c=0.53, same but quadratic log-sd on the left: 0.1948
   - c=0.53 with log-linear sd: 0.1956; c=0.50: 0.1967
   The standard error of a CV mean is about 0.025, but differences between candidates are
   paired and much smaller.

## Final program
Piecewise with c=0.55: a quartic mean with log-linear sd on the left; on the right a mixture
of two quadratic branches (weights 0.514/0.486) with a shared log-linear sd. It uses
18 constants. I refit it on all the data. In-sample NLL is 0.181 and repeated 10-fold CV
gives 0.1946. I chose it because it is the simplest model that ties for the best CV score.
To allow for the small optimism from selecting among about 60 configurations, I add a small margin.

expected_nll: 0.20
