# Notes

## Data
1000 rows, x0 in [0,1], y in [-1.7, 2.7]. Scatter plot: for x < ~0.6 a single noisy curved
branch (min ~-0.75 near x=0.2, rising to ~0.3 at x=0.6). Around x≈0.65 it splits into two
branches: an upper one rising almost linearly to ~2.5 at x=1, and a lower one curving down
to ~-1 at x=1. Noise shrinks with x (sd ~0.3 at the left, ~0.1 at the right). Binned stats show
strong bimodality (kurtosis ≈ -2) for x > 0.7.

## Model
Two-component mixture of Normals:
- lower mean m1 = a + b x + A sin(w x + phi) (one sine wave covers the dip and the turn-down);
- upper mean m2 = m1 + c * s * softplus((x - x0)/s): identical to m1 before the split, then
  diverges linearly (smooth hinge, x0 ≈ 0.667);
- log sd of each component linear in x; upper-branch weight = sigmoid(linear in x).
14 fitted constants (+ literal 1.0s), fitted by maximum likelihood (Nelder–Mead then BFGS, restarts).

Also tried: same structure with Student-t components (shared df) — df went to ~900, i.e.
Normal; CV slightly worse (0.2220 vs 0.2212), so Normal kept.

## Validation
5-fold CV (random permutation, seed 0), refitting all parameters on each training split,
warm-started from the full-data fit:
- Normal mixture: CV NLL 0.2212 (in-sample 0.2095)
- Student-t mixture: CV NLL 0.2220

Final model is the Normal mixture refit on all data. Expected held-out NLL is the CV value
with a small margin for selection/optimism.

expected_nll: 0.225
