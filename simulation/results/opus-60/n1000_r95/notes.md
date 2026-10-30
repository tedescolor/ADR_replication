# Notes

## Data
1000 rows, x0 in [0,1]. Scatter shows one noisy curved branch (sd ~0.3) for x<~0.5
that splits into two branches: a lower one (rises to ~0.2 at x~0.65, then falls to -1 at x=1,
sd shrinking to ~0.08) and an upper, nearly linear one (0.4 at x=0.6 -> 2.5 at x=1, sd ~0.14).
Above x~0.65 about half the points are on each branch.

## Model family
Two-component Normal mixture, u = 2*x0 - 1:
- lower: mean poly(u, deg a), log-sd poly(u, deg 2)
- upper: mean poly(u, deg b), log-sd linear in u
- upper weight w = wmax * sigmoid(k*(x0 - x_c))
Fitted by maximum likelihood (BFGS, several jittered starts).

## Validation
5-fold CV, repeated with 2 different shuffles (refit on each training fold, starting from the
full-data fit). Mean held-out NLL:
- first attempt (raw-x polys, linear-logit weight; upper comp leaked at low x): 0.263-0.267
- capped-sigmoid weight, (a,b,sdA,sdB) = (5,1,2,0): 0.257; (7,1,2,1): 0.250; (7,2,2,1): 0.246
- (7,3,2,1): 0.241; (7,4,2,1): 0.2380; (7,5,2,1): 0.2374
- Student-t instead of Normal: no gain (0.2458 vs 0.2455); sin-based lower mean: worse (0.26)
- constant 50/50 weight everywhere: worse (0.244)
- higher lower-mean degree (8, 9): worse

## Choice
(7,4,2,1) Normal: 21 fitted constants (~25 literals), CV 0.238, essentially tied with
(7,5,2,1) but simpler. Train NLL 0.215. Allowing a small optimism for model selection
over ~20 configs, estimate for new data ~0.243.

expected_nll: 0.243
