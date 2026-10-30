# Notes

## Data
250 rows, x0 in [0,1]. Scatter shows two branches:
- a smooth wiggly curve: about -0.2 at x=0, a dip to about -0.85 at x=0.2, a peak of about 0.2 at x=0.68, then
  down to about -0.9 at x=1. It is noisier at small x (sd about 0.45) than at large x (sd about 0.1).
- a straight line, y ≈ 5x - 2.4, with sd shrinking from about 0.4 to 0.1. It overlaps the curve for 0.2<x<0.6
  and separates from it above 0.6. It is absent below x≈0.21, where it would sit at y < -1.4 and the data has no points.

## Validation
5-fold CV, repeated with 3–4 different fold permutations. Every fold is refit from data-driven
initialisations (polyfit on a rough branch split), so the test fold is not used for fitting. Reported numbers are mean held-out NLL.

## What I tried (CV NLL)
- Generic 2-normal mixtures with polynomial means, log-sd and logit weight, random inits: 0.35–0.42. Unstable, with local optima.
- Structured: comp1 = poly mean, comp2 = line, weight of comp2 = wmax*sigmoid(k(x-x0)):
  - two optima: x0≈0.22 (line present from 0.22 on) and x0≈0.6 (line only in the split region).
    Fitting each mode separately in CV gave x0≈0.22 at 0.309 and x0≈0.6 at 0.347, and averaging the two gave 0.312.
    The x0≈0.22 mode was better in every repeat.
  - within that mode: quartic mean, log-linear sds, wmax fixed at 0.5 → **0.304** (13 constants).
    The variants were all as good or worse: free wmax 0.308, quadratic log-sd 0.306, cubic mean 0.323, quintic mean 0.313,
    quadratic line 0.314, constant line sd 0.338, sine+linear mean 0.314.
  - Student-t components: no gain (0.339–0.342 when mode selection was free).

## Final program
Mixture of two normals:
- Normal(quartic(x), exp(a+bx)) with weight 1-w;
- Normal(line(x), exp(c+dx)) with weight w = 0.5*sigmoid(73.55(x-0.2129)).

13 constants. Train NLL is 0.237 and repeated CV NLL is 0.304 (range 0.289–0.317 across fold permutations).
The final fit uses all 250 rows, which should help slightly, but choosing the best of several variants biases the CV figure low.
So I estimate a bit above the CV value.

expected_nll: 0.31
