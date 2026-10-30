# Notes

## Data
1000 rows, x0 in (0,1), y. Scatter shows one noisy band for x < ~0.52 (mean dips to -0.75
near x=0.25, sd ~0.35, mildly heavy tails), then it splits into two branches of about equal
weight: an upper branch rising roughly linearly to ~2.5 at x=1, and a lower branch flat near
0.1 until ~0.8 then falling to ~-1. Noise shrinks with x (sd ~0.1 at x=1).

## Model family
2-component Gaussian mixture with constant weight:
- upper mean mA(x): polynomial;
- lower mean mB = mA - gap(x), gap = h*(c0 + c1 h + c2 h^2), h = softplus hinge at x≈0.52
  (fit made it nearly a hard hinge), so both components coincide before the split;
- log-sd of each component: quadratic in x. Before the split the two components act as a
  scale mixture (heavy-ish tails), after it they are the two branches.
Fitted by maximum likelihood (BFGS + Powell) with scipy.

## Validation
5-fold CV (fold refits warm-started from the full fit), several seeds.
| variant | CV NLL |
|---|---|
| dA=5 poly, linear hinge gap, dS=2 | 0.2446 |
| dA=6, linear gap | 0.2433 (seeds 1,2: 0.2472, 0.2498) |
| dA=7, linear gap | 0.2428 |
| dA=6, Student-t components | 0.2424 (nu -> inf on full data, i.e. Normal) |
| dA=6, logistic weight in x | 0.2455 |
| dA=6, cubic gap | 0.2388 |
| **dA=5, cubic gap (final)** | 0.2386 (seeds 1,2: 0.2418, 0.2510; mean 0.2438) |
| dA=4, cubic gap | mean over seeds 1,2: 0.2477 |
| shared sd + Student-t | 0.2398 / 0.2410 |
| dA=5, dS=3 separate sd | 0.2418 |

Seed-to-seed CV variation (~0.01) is larger than most differences; the cubic-gap family was
consistently a bit better than the linear gap. Final chose dA=5, dS=2, cubic gap: best mean CV,
18 fitted parameters (~22 literals in model.py). Training NLL 0.2259.

## Estimate
Mean CV over 3 seeds ≈ 0.244; slightly padded for selection optimism.

expected_nll: 0.246
