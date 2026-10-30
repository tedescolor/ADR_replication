# Notes

## Data
1000 rows, x0 in [0,1]. Scatter shows a single noisy curve (sinusoid-like, min ~-0.8 at x~0.2)
for x < 0.5, which splits at x ~ 0.5 into two branches: an upper branch rising to ~2.5 at x=1
and a lower branch peaking ~0.1 near x=0.7 and falling to ~-0.9 at x=1. Noise decreases with x
(sd ~0.35 near 0, ~0.12 on the branches near 1). Branches hold roughly 50/50 of the mass.

## Validation
5-fold CV (mean held-out NLL), repeated with 2 shuffles for the final comparisons; multiple
random restarts per fit (BFGS then Powell). Fold-to-fold spread ~0.03, so differences < ~0.003
treated as noise.

## Candidates (CV NLL)
- 2-normal mixture, free polynomial means / log-sd per component, const or logistic weight:
  best 0.2475 (quartic means, quadratic log-sd); higher degrees overfit (up to 0.267).
- "Split" model: shared cubic base curve; each branch = base + polynomial in a ramp
  r = softplus/relu(x - c); the fitted softplus width went to ~0, i.e. a hard kink at c=0.498.
  - soft ramp, linear log-sd: 0.249; quadratic log-sd: 0.2446; Student-t tails: no gain.
  - hard kink (relu), quadratic branch offsets, quadratic log-sd per branch, constant weight:
    0.2452 (2x5-fold), 16 parameters. Shared sd: 0.2474; cubic branch offsets 0.2495;
    quartic base 0.2471; cubic or linear log-sd 0.2478 / 0.2504.
PIT values of the split model are uniform across x bins (no clear misfit).

## Final
Hard-kink split model (3,2,2): mixture of two normals sharing a cubic base curve, branches
diverge as quadratics in max(x-0.498,0), each with quadratic-in-x log-sd, constant weight 0.536.
16 fitted constants (17 literals incl. the 0.0). Train NLL 0.2296, CV 0.2452.
Estimate for new data: CV value (fit on 80% data; full-data fit slightly better, small
selection optimism among ~10 candidates roughly cancels).

expected_nll: 0.245
