# Notes

Data: 250 rows, x0 in (0, 1). The mean dips (about -0.8 near x=0.3) and then rises. For x > ~0.6
y splits into two narrow branches: an upper one rising to ~2.5 at x=1 and a lower one falling
to ~-1. Noise is ~0.3 at low x and ~0.12 on each branch at high x.

Model family: a 2-component Gaussian mixture. Each component mean is a polynomial in x of
degree dm, each log-sigma is a polynomial of degree ds, and the mixing logit is a polynomial
of degree dw. Fitted by maximum likelihood with BFGS and several restarts.

Validation: K-fold CV on held-out NLL (folds from a fixed permutation, refit within each fold).
- (1,0,0) 10-fold: 0.624
- (2,1,0) 10-fold: 0.492
- (3,1,1) 5-fold: 0.278 (train 0.197)  <- chosen
- (3,2,1) 5-fold: 0.303 (train 0.181)
- (3,2,2) 5-fold: 0.318 (train 0.190)

Final: (3,1,1), fitted on all data. It has 14 fitted constants plus the literal 1.0 in the
weights (15 in total). Two components with cubic means, log-linear sigmas and a logistic-linear
weight. At low x the fitted means are close (about -0.8 and -0.5 at x=0.3) and the components
overlap, so the mixture forms one broad mode there. At high x they separate into the two branches. I chose it because it had the best
CV, and the richer variants overfit. My estimate is the CV value plus a small margin for
selection among about 5 candidates.

expected_nll: 0.29
