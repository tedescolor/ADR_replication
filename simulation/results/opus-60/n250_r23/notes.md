# Notes

## Data
250 rows, x0 ~ U(0,1). Scatter plot: one noisy curve for x < ~0.55 (dips to -0.9 at
x~0.25, rises to ~0.3 at 0.6). Past ~0.55 it splits into two branches. The lower branch
follows the same smooth curve and turns down to -0.9 at x=1. The upper branch rises to
~2.3. The gap between the branches grows roughly linearly from the split point. Noise is
larger at small x (sd ~0.35) than at large x (sd ~0.1).

## Final program (16 constants)
A two-component normal mixture with constant weights (0.455 / 0.545):
- lower branch mean m(x) = degree-6 polynomial
- upper branch mean m(x) + (a + b*h)*h, with h = max(x - c, 0)
- shared sd exp(s0 + s1*x)
Below the split point both components are identical, so the density is one normal there.

## Validation
5-fold CV repeated 4 times (20 fits; the model is refit on each training split with
several BFGS restarts). Mean held-out NLL per row, with SE over folds of ~0.016:
- free 2-component mixture, separate poly means, sigmoid weight: 0.09 to 0.16
- shared curve + linear hinge jump, base = sinusoid+linear: 0.110; poly5: 0.106; poly6: 0.056; poly7: 0.056
- poly6 variants: constant sd 0.156, quadratic log-sd 0.057, separate sd per branch 0.065,
  x-dependent weight 0.059, Student-t 0.0565
- **poly6 + quadratic hinge jump: 0.039** (chosen; best, and still compact)
Training NLL of the final fit: -0.009.

## Estimate
The CV fits used 80% of the data, so the full-data fit should do a bit better. But the
model was picked from about 15 candidates, which biases the CV score low. These roughly
cancel, so I add a small margin.

expected_nll: 0.05
