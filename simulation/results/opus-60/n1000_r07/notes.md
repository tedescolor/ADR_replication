# Notes

## Data
1000 rows, x0 in [0,1], y in [-1.6, 2.7]. A scatter plot shows two curved branches.
On the left (x<0.5) they overlap and look like one noisy curve. After x~0.55 they split:
one rises almost linearly to 2.6 and the other bends down to -1. The noise shrinks with
x (sd ~0.38 at x=0, ~0.09 at x=1). Within the branches the residuals look Gaussian:
a Student-t fit sent df to about 9500.

## Validation
5-fold CV repeated twice (10 fits per candidate). Each fit used BFGS MLE with several
random restarts, started from the full-data fit. The score is the mean held-out NLL
in nats. I used t = 2*x0 - 1 to keep the optimisation well conditioned.

## Candidates (CV NLL / train NLL / #params)
- 2-comp normal mixture, poly means, log-linear sds, logit-linear weight:
  deg(4,4,1,1,1) 0.2250/0.2100/16; deg(5,5,2,2,2) 0.2344/0.1988/21; others 0.22-0.24.
  The fitted weight was about 0.5 and constant in x.
- constant weight: (4,4,1,1,0) 0.2233/15. Constant sds: 0.290 (much worse).
- sd linear in x instead of log-linear: 0.2220/15. Shared between branches, weight
  fixed at 0.5: 0.2206/12. Quadratic sd: 0.2211. Cubic means: 0.2285. Quintic: 0.2239.
- Student-t components: no gain.
- Center/gap parameterisation, means c(t) +/- g(t):
  - c quartic, g = t(b1+b2 t): 0.2206/9
  - with a free weight: 0.2211
  - quadratic sd: 0.2214
  - quintic c: 0.2210
  - sine center: 0.2337 (worse)
  - **c without the t^2 term: 0.2192 / train 0.2120 / 8 params (chosen)**

## Final program
Mixture of two Normals with weight 0.5 each:
- means: c(t) +/- g(t), where c(t) = a0 + a1 t + a3 t^3 + a4 t^4 and g(t) = t (b1 + b2 t)
- shared scale: |s0 + s1 t|
- 8 fitted constants (13 literals in total)

It has the best CV score and the fewest parameters. The other top candidates are
within about 0.002 of it, so selection bias is small.
Estimate: CV 0.219 plus a small allowance for selection.

expected_nll: 0.222
