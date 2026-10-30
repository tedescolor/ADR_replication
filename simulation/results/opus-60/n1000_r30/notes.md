# Notes

## Data
1000 rows, x0 in [0,1]. Scatter: a single noisy curve for x < ~0.4 (dip to about -0.85 at x=0.25),
which then splits into two branches: an upper, almost linear branch rising to ~2.5 at x=1, and a
lower branch that is flat around 0.1 and then drops to -1 at x=1. Within a branch the noise is
roughly Gaussian with sd ~0.3 at x=0, shrinking to ~0.1 at x=1.

## Candidates (5-fold CV, same folds, mean held-out NLL)
A) Symmetric 2-component Gaussian mixture: polynomial means of degree dm in (x-0.5), log-sd
   polynomials of degree ds, logit weight polynomial of degree dw in x. MLE with BFGS and
   several restarts.
   - (3,1,1): train 0.173, CV **0.186**
   - (5,2,1): train 0.157, CV 0.194
   - (5,2,3): train 0.152, CV 0.192
   - (7,2,3): train 0.149, CV 0.196
B) "Branching" mixture: a lower branch over the whole range, an upper branch switched on by
   w = sigmoid(a)*sigmoid(k(x-c)), with separate polynomial means and log-sds.
   - (5,1,1,1) CV 0.193; (6,2,2,1) CV 0.191; (7,2,2,2) CV 0.194; (8,3,3,2) CV 0.190
   The fitted switch was a hard step near x=0.38, where the two branches cross.

Every variant lands at 0.186–0.196. The differences are about the size of the CV noise, and
the more flexible models overfit (the gap between train and CV NLL grows). Runs of shared-sd
variants of B were lost because their background processes were killed.

## Final
Model A (3,1,1): a mixture of two Normals with cubic means, log-linear sds and a logistic-linear
weight; 14 fitted parameters (16 literals counting the 0.5 centring and the 1 in 1-w),
refit on all 1000 rows. It has the best CV score and is also the simplest model. Train NLL is
0.173 and 5-fold CV NLL is 0.186. I add a small allowance for selection optimism.

expected_nll: 0.19
