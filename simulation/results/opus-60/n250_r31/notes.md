# Notes

## Data
250 rows, x0 in [0,1], y in [-1.4, 2.5]. A scatter plot shows two branches:
- a main branch present for all x: near 0 at x=0, dips to about -0.85 at x≈0.25, then rises almost
  linearly to about 2.5 at x=1. It is wide (sd≈0.3–0.4) on the left and tight (sd≈0.1) on the right;
- a second, lower branch that shows up from x≈0.5: about 0.3 at x≈0.65, curving down to about -0.9 at x=1.
  It is tight and carries roughly 20–35% of the mass on the right.
So the conditional distribution is bimodal for x>0.6, and a single Gaussian cannot fit it.

## Candidates (validation: 5-fold CV, repeated 1–3 times with different shuffles; restarts of L-BFGS
for each fold; mean held-out NLL ± s.e. over folds)
Notation: polynomials in t=2x-1; mN = degree of the mean, sN = degree of the log-sd, wN = degree of the mixing logit.
- K1 Normal, m5 s3: CV 0.673
- K2 m3,2 s1,1 w1 (13 params): CV 0.306 (1 rep), 0.306 ± 0.020 (3 reps)
- K2 m3,2 s1,0 w1: 0.330; K2 m3,2 s2,1 w1: 0.305; K2 m3,2 s1,1 w2: 0.299 / 0.312
- K2 m3,2 s2,1 w2: 0.299; K2 m3,2 s2,0 w2: 0.326; K2 m4,2 s1,1 w1: 0.308
- K2 m4,3 s2,1 w2: 0.317; K2 m5,3 s2,2 w2: 0.431 (overfits/unstable)
- K3 with an extra broad constant component: 0.331; Student-t components: 0.320
All the reasonable 2-component models are within CV noise of each other (~0.30–0.31).

## Choice
Picked the simplest of these: K2 m3,2 s1,1 w1 (13 constants). It has a cubic-mean main branch with
log-linear sd, a quadratic-mean second branch with log-linear sd, and a logistic-linear weight for the
second branch (weight ≈0.02 at x=0 and ≈0.33 at x=1). It is refit on all the data (train NLL 0.244).
Bigger models did not improve CV reliably.

## Expected NLL
CV ≈ 0.306 with models trained on 80% of the data; the full-data fit should be slightly better,
but the train/CV gap and fold noise (s.e. ≈0.02–0.03) argue for a conservative estimate.

expected_nll: 0.31
