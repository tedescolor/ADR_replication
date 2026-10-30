# Notes

## Data
250 rows, x0 in [0,1]. A scatter plot shows two branches: they overlap (or are very close)
for x < ~0.5 and then split. The upper branch rises almost linearly to ~2.4 at x=1, and the lower
one turns back down to ~-0.9. The noise shrinks with x (residual sd is ~0.4 near x=0 and
~0.1 near x=1). Standardised residuals are close to Gaussian (excess kurtosis -0.36, skew ~0).

## Validation
Repeated K-fold CV: one 5-fold split plus one 10-fold split, averaged (each point is held out
twice). Each fit is a full-likelihood fit (BFGS, several random restarts) on the training
folds only. Scripts are in `work/fit*.py`.

## Candidates (CV mean NLL)
- 2-component Gaussian mixture with free polynomial means (deg 3–5), log-linear sds,
  constant or sigmoid weight: 0.20–0.24 (5-fold).
- Shared base cubic g, branches g + a·h and g − b·h, with h = a softplus ramp at c:
  separate sds 0.184, shared sd 0.180, shared sd with a learnt weight 0.184, quadratic
  log-sd 0.182.
- Lower branch = sinusoid + line, with a softplus split: 0.182. Power-law split D·x^p: 0.181–0.184.
  Hard ReLU split: 0.197.
- Quartic base with a softplus split: 0.176 (10 params; this fit is sensitive to the starting point).
- **Cubic g, branches g and g + D·softplus((x−c)·k)/k, equal weights, shared sd
  exp(q0 + q1·x): 0.173** (9 fitted params, train NLL 0.140).

## Choice
The last model has the best CV and the fewest parameters. Its training NLL is the same as the
10-parameter version, so the dropped parameter was redundant. The final program has
12 literals and 125 AST nodes, and its parameters were refit on all 250 rows.

## Estimate
The CV of the chosen model is 0.173. I add a little for optimism from choosing among about
10 variants. Fitting on 100% of the data instead of 80–90% pushes slightly the other way.

expected_nll: 0.18
