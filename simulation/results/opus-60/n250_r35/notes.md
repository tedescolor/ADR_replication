# Conditional density model notes

## Data
250 rows, x0 in (0,1). Below x≈0.6, y follows one noisy curve (dip to about -0.85 near
x=0.22). Above x≈0.6 it splits into two branches: one rises steeply (to about 2.5 at x=1)
and the other falls (to about -1 at x=1), with roughly 55/45 of the points split
upper/lower. The noise SD shrinks with x, from about 0.32 near x=0 to about 0.12 near x=1.

## Validation
5-fold CV repeated twice (fixed seeds, same folds for every model), refitting by ML
(BFGS + Nelder-Mead, 4 restarts) on each training split. Score = mean held-out NLL.

## Candidates (CV NLL)
- Main polynomial curve + gated linear upper branch (sigmoid gate), deg 3–6, log-linear/quadratic sd: best 0.269 (deg5, 15 params)
- **Hinge split** (chosen): trend f(x) quartic; softplus hinge h(x)=s*log(1+exp((x-c)/s));
  branches f+b1*h and f-b2*h; shared sd exp(a+b x); constant weight. 12 params: CV **0.2647** (train 0.196)
- + separate sd per branch (14): 0.280; quadratic log-sd (13): 0.270; x-dependent gate (13): 0.2645 (no real gain)
- same hinge split with a cubic trend (11 params): **0.2574** (train 0.199); quintic trend (13): 0.278

Final choice: the hinge split with a cubic trend. It has the fewest parameters and the
best CV score. In the fit, the cubic follows the rising branch, and the falling branch
is f - 8.86*h. The softplus is written in an overflow-safe form. The program has 15
literal constants (the 11 fitted parameters plus 0.0/1.0 helpers and repeats).

## Estimate
CV gives 0.257. The CV scores of near-equivalent models vary by about ±0.01, and the
train/CV gap is about 0.07, so I add a small margin for selection optimism.

expected_nll: 0.27
