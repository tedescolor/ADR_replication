# Notes

## Data
1000 rows, x0 in [0,1], y continuous. Scatter shows a single noisy curve for x < ~0.5
(sd ~0.4 at x=0 shrinking to ~0.2 at x=0.5), which splits (pitchfork) into two
branches for x > ~0.5: an upper branch rising to ~2.5 at x=1 and a lower branch falling
to ~-1, roughly 50/50 in weight, each with small noise (~0.1 near x=1).

## Model family
Two-component Gaussian mixture, fitted by maximum likelihood (BFGS, several starts):
- component A: mean = polynomial in t = x-0.5, log-sd = polynomial in t;
- component B: mean = polynomial in t, log-sd polynomial (or shared: sdB = c * sdA);
- weight of B = sigmoid(a) * sigmoid((x - c) * exp(s)) (switches on at the split).

## Validation
5-fold CV (2-4 repeats with different shuffles), refitting everything inside each fold;
paired fold-by-fold differences used to compare candidates.

| variant (degA mean / logsdA / degB mean / logsdB) | train NLL | CV NLL |
|---|---|---|
| 5/2/3/1 | 0.2176 | 0.2391 |
| 6/3/3/1 | 0.2175 | 0.2414 |
| 7/3/3/1 | 0.2161 | 0.2394 |
| 6/4/3/2 | 0.2132 | 0.2389 |
| 6/3/2/1 | 0.2199 | 0.2422 |
| 8/4/3/1 | 0.2157 | 0.2416 |
| 5/1/3/1 | 0.2179 | 0.2393 |
| 5/2/3/2 | 0.2137 | 0.2359 |
| 5/2/3/2 Student-t | 0.2128 | 0.2376 |
| shared sd 5/1/3 | 0.2179 | 0.2368 |
| shared sd 5/2/3 | 0.2171 | 0.2382 |
| shared sd 6/2/3 | 0.2170 | 0.2402 |

Paired CV (4x5 folds, same splits): shared 5/1/3 = 0.2366; quadratic B mean +0.0034 (+-0.0017)
worse; constant sd +0.088 worse; separate sd 5/2/3/2 +0.0002 (+-0.0014); quartic A mean
(shared 4/1/3) = 0.2353, -0.0013 (+-0.0007) better. All reasonable variants are within ~0.005
of each other (standard error of a CV mean ~0.012), so the simplest best one was chosen.

## Final program
Shared-sd 4/1/3 model: quartic mean for A, cubic for B, log-linear sd shared by both
branches (B scaled by a constant), smooth sigmoid onset of B around x=0.48 with overall
B weight ~0.54. 15 fitted constants + 2 structural (0.5 centring, 1 in 1-pB) = 17 literals.
Train NLL 0.2184, paired-CV NLL 0.2353. The final fit uses all 1000 rows (vs 800 in CV), so
the true held-out NLL should be about the CV value or marginally lower.

expected_nll: 0.235
