# Notes

## Data
- y lies in [-0.07, 7.08]. About 81% of rows sit at a small set of discrete values plus
  Gaussian noise with sd ~0.020: 0, 0.5, 1, 7/6, 11/6, 2, 3, 4, 4.5, 5, 6, 7 (0 alone is 34%).
  The other ~19% form a continuum, mostly on 2.05–3.9, with a thin tail up to ~6.5.
- The binary covariates are 0/1 plus small jitter. x0 and x1 are almost identical, and so are x11 and x17.
  x5, x6 and x20 are skewed, so I used them on a log scale.
- Covariates carry modest information: in a 13-class multinomial logit on the "which atom /
  continuum" label, CV log-loss went from 2.076 (marginal) to 1.989. A rank-3 low-rank logit did
  slightly better (1.985).

## Model family
Mixture of narrow normals at the discrete values (one shared sd) plus 4 broad normals for the
continuum. Every component's log-weight is a_k + b_k*s + c_k*t, where s and t are two linear
scores of the covariates (a reduced-rank multinomial gate). All parameters are fitted jointly by
maximum likelihood on the real y density (L-BFGS, analytic gradients, small L2 penalty).

## Validation
5-fold CV (fixed random folds), mean held-out NLL on y:
- unconditional mixture (13 atoms + 4 broad): 0.296
- rank-1 gate, all 24 features: 0.200
- rank-2 gate, all 24 features: 0.165 (needs about 120 constants, so it's over budget)
- rank-2 gate with the continuum as one gated group (shared internal weights): ~0.20 train, worse.
  The broad components need their own gates.
- rank-2 gate, 8 features: 0.170; 6 features and 13 atoms: 0.1698
- **final**: rank-2 gate with 7 features (log x5, log(|x6|+1), x22, x19, x0, x3, x15), 12 atoms
  (5/3 dropped) and 4 broad normals: **CV 0.1689** (fold NLLs 0.180, 0.123, 0.178, 0.213, 0.152;
  standard error over rows ~0.019). Fitting on all data gives a training NLL of 0.156.
  The exported program reproduces the fitted density: rounding to 4 significant digits changed it by <1e-4.

I chose the final model as the best CV under the 80-constant / 500-node limit (it uses exactly
80 / 500). The reference class (atom 0) has log-weight 0, and the two scores are scaled so that one
broad component has unit loadings. This frees constants for the fit.

## Estimate
The CV NLL is 0.169. The final model is refit on all the data, which should help slightly, but
there is some selection optimism from comparing a few variants. I estimate ~0.17.

expected_nll: 0.17
