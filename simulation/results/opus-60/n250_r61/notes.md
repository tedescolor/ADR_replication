# Notes

## Data
250 rows, x0 in (0,1), y in about (-1.3, 2.5). A scatter plot shows a single noisy curve for
x < ~0.55: it dips to about -0.8 near x=0.2 and rises to about 0 at x=0.5. Past that point it
**bifurcates**. The upper branch rises almost linearly to about 2.4 at x=1. The lower branch
peaks around 0.2 near x=0.7 and falls to about -0.85 at x=1. Noise shrinks as x grows (sd
about 0.4 at x=0, about 0.1 at x=1).

## Validation
5-fold CV repeated twice (10 fits, 500 held-out predictions), measured as mean held-out NLL.
Each fold is refit by BFGS from several starts (the full-data optimum plus jittered copies) and
the start with the best training loss is kept.

## Candidates (train NLL / CV NLL)
- A: 2-component Gaussian mixture. Lower mean is a polynomial, upper mean is linear, logit
  weight and log-sds are linear in x. deg3 0.242/0.288, deg4 0.230/0.279, deg5 0.227/0.292.
  Student-t components: nu went to about 8 with no gain (0.288).
- B: "bifurcation" mixture. Lower mean is a polynomial m1. Upper mean is
  m1 + k*relu(x - x0). Weight is w*1[x > x0]. Log-sds are linear in x.
  deg3 0.235/0.273, **deg4 0.199/0.242**, deg5 got stuck in a worse optimum (0.259).
  t-noise gave no gain.
- B4 with the weight logit linear in x: 0.194/0.240. Constant upper sd: 0.218/0.267 (worse).
- **C: B4 with a power onset, k*relu(x - x0)^p: 0.195/0.236** (p is about 1.13, x0 about 0.545).
  The deg-5 version of C: 0.193/0.242.

## Final
Model C: a Gaussian mixture with weights (1-w, w), where w = 0.571 for x > 0.545 and 0 otherwise.
- Lower component: quartic mean, with a log-linear sd.
- Upper component: the quartic plus 8.26*(x-0.545)^1.128, with a log-linear sd.

This model uses 17 literal constants. It had the best CV score and is consistent with the
physical picture of one curve splitting into two branches. The CV differences between B4, E
and C are small (about 0.005), roughly the size of the CV noise.

Estimate of held-out NLL: CV gave 0.236. I add a little for selection among about 12
candidates and for the instability of the change point.

expected_nll: 0.245
