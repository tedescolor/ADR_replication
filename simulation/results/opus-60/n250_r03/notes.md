# Notes

## Data
250 rows, one covariate x0 in [0,1]. Scatter plot: for x < ~0.6 there is a single noisy
branch (it dips to about -0.75 at x~0.25 and then rises). Around x~0.6 it splits into two thin
branches: an upper one rising to ~2.5 at x=1 and a lower one falling to ~-0.9. The two branches
hold roughly equal numbers of points. The noise sd shrinks steadily with x (about 0.47 near x=0,
about 0.08 near x=1), and the residual kurtosis is about 0, so Gaussian noise is fine.

## Candidates (5-fold CV, repeated 2-3 times, mean held-out NLL per row)
- Free 2-component Gaussian mixture, cubic means, log-linear sds, logistic weight (14 params): CV 0.303
  (quartic means / quadratic log-sd: 0.32-0.35, overfit).
- "Pitchfork" model: shared cubic f(x); upper mean = f + a*softplus_k(x-c), lower mean = f - b*softplus_k(x-c);
  constant mixture weight; shared sd(x), with a multiplicative shift for the lower branch after the split.
  - log-linear sd (12 params): CV 0.296
  - log-quadratic sd (13): 0.299
  - sd = softplus(linear) (12): **0.293 ± 0.011**  <- chosen
  - quartic f (13): optimisation unstable, worse; separate shift for upper branch (13): 0.304;
    Student-t noise (13): 0.299; logistic weight: no gain.
- Fold refits got stuck in bad local optima (early CV numbers of 0.4-0.5). Final CV runs use multi-start
  BFGS -> Nelder-Mead -> BFGS, with the full-data optimum as one extra starting point. Only the starting
  point comes from the full data, so the leak is small.

## Final program
Pitchfork model, Gaussian components, 12 fitted parameters (17 literals in total). Train NLL 0.238, CV NLL 0.293.
Several variants were compared, so the CV minimum is slightly optimistic; I add a small margin.

expected_nll: 0.30
