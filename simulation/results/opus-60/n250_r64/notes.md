# Notes

## Data
250 rows, x0 in [0,1]. Scatter plot: a single noisy curve for x < ~0.6 (dips to about -0.75 near x=0.25,
rises to about 0.2 near x=0.6). Above ~0.63 it splits into two branches: one keeps rising
almost linearly to about 2.6 at x=1 (about 60% of points), the other bends down to about -0.9 at x=1.
Noise sd falls steadily with x (about 0.4 at x=0, about 0.12 at x=1). Residuals are not heavy-tailed.

## Validation
Repeated 5-fold CV (4 repeats with fixed seeds, the same folds for every model), refitting all parameters
by ML (Nelder-Mead then BFGS, warm-started) on each training split. I report the mean held-out NLL.

## Candidates (CV NLL)
- A: sine trunk + an independent linear upper branch with a sigmoid weight in x: 0.310 (1 repeat)
- B: sine trunk + upper branch = trunk + a*softplus((x-c)/h), constant weight, log-linear sds: about 0.26 (2 repeats: 0.269, 0.251)
- B variants (log-quadratic sd, quadratic branch, weight slope): no better. Student-t noise: df goes to infinity (normal).
- Two branches both diverging from a shared trunk (sine or cubic or quartic trunk), separate sds: 0.258-0.263
- Same with a shared sd: 0.254
- **SZ: cubic trunk = rising branch; falling branch = trunk - k*softplus; shared log-linear sd; constant weight: 0.2472**
  (repeats 0.249, 0.238, 0.237, 0.265). Train NLL 0.203.
- Adding an extra high-frequency sine to the trunk: 0.282 (overfits). Quadratic term on the branch: 0.253.

## Final program
2-component Normal mixture. m1 = cubic in x. m2 = m1 - k*h*softplus((x-c)/h). Shared sd = exp(a+b*x).
Constant weight (0.40 on the falling branch). 10 fitted constants (13 literals including the constants
1 and 2 in the stable softplus). I chose it because it has the lowest CV and is also the simplest model.

The CV estimate is 0.247. The final fit uses all the data, which helps a little. Picking the best of about
12 models adds some optimism, and the spread across fold repeats is about ±0.013. So my estimate is slightly above CV.

expected_nll: 0.255
