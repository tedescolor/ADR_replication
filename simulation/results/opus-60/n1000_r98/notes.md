# Notes

**Data.** There are 1000 rows with x0 in [0,1]. Below x≈0.5, y follows a single smooth curve with
noise sd≈0.3: it dips to about -0.75 near x=0.25 and then rises. Above x≈0.5 the data splits
into two branches of roughly equal size. The upper branch keeps rising to y≈2.5 at x=1, and
the lower branch bends down to y≈-1. The noise shrinks as x grows (sd≈0.12 at x=1), and the
residuals look Gaussian (skew and kurtosis are close to 0).

**Validation.** I used 5-fold CV with one fixed permutation. Every fold was fitted by maximum
likelihood (BFGS, several jittered restarts) on the training folds and scored by mean NLL on
the held-out fold. I compared models on CV NLL.

**Tried (CV NLL):**
- Free 2-component mixture with a sigmoid gate, poly4 mean, separate lower-branch poly (16 params): 0.190
- Symmetric branches f ± g(relu(x-c)), poly3/poly4 f, quadratic log-sd: 0.210 / 0.189
- Symmetric branches with f as a sine plus a line: 0.210
- Asymmetric branches f+g1, f-g2: 0.187 (also tried a fitted weight, a cubic log-sd and poly5; none helped)
- Upper = poly4 f, lower = f - (h2 r^2 + h1 r) with r = max(x-c, 0), quadratic log-sd: 0.1851
- Same model with **linear log-sd (10 params): 0.1845** (chosen)
- Variants of it: h = k r^2 only, 0.192; h = k r^p, 0.186; separate sd per branch, 0.190; Student-t, 0.187

**Final program.** An equal-weight mixture of two normals that share one sd,
exp(a + b x). Upper mean: a quartic in x. Lower mean: the upper mean minus
(h1 r + h2 r^2), with r = max(x - 0.50, 0). Below the knot the two components are
identical, so the model is a single normal there. It has 10 fitted constants (13 numeric
literals, including the two 0.5 weights and the 0.0 in maximum). I chose it because it had
the best CV score and is the simplest of the competitive models. In-sample NLL is 0.172.

expected_nll: 0.186
