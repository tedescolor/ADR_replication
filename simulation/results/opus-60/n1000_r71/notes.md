# Notes

## Data
1000 rows, x0 in (0,1), y in about (-2, 2.7). A scatter plot shows a single wide noisy branch for
x < ~0.5 (sd ~0.35–0.45) that splits into two narrow branches (sd ~0.1–0.15) for x > ~0.55:
one rises roughly linearly to ~2.5 at x=1, the other peaks near 0.1 at x~0.7 and falls to ~-1.
The split is close to 50/50. Noise shrinks as x grows. Residuals within each branch are close to
Gaussian, with mild excess kurtosis.

## Model family
A 2-component mixture with z = 2*x0 - 1:
- each component Normal(poly_mean(z), exp(poly_logsd(z)));
- mixing weight sigmoid(poly(z)).
Fitted by direct maximum likelihood (BFGS with several random restarts).

## Validation
5-fold CV repeated twice (2 different shuffles), refitting on every training split and warm-starting
from the full-data fit. The reported number is the mean held-out NLL over the 2000 held-out
predictions. The SE of the mean is ~0.024, and differences between variants on the same folds are much smaller.

| spec (deg mean A, logsd A, mean B, logsd B, weight; family) | k | train | CV |
|---|---|---|---|
| 3,1,3,1,1 N | 14 | 0.2296 | 0.2442 |
| 3,2,3,2,2 N | 17 | 0.2190 | 0.2476 |
| 4,2,4,2,2 N | 19 | 0.2159 | 0.2463 |
| 5,2,5,2,2 N | 21 | 0.2143 | 0.2447 |
| 4,2,4,2,2 Student-t | 20 | 0.2159 | 0.2465 |
| 5,2,5,2, fixed w=0.5 N | 18 | 0.2268 | 0.2511 |
| 5,2,5,2,0 N | 19 | 0.2239 | 0.2484 |
| 3,2,4,2,1 N / t | 17/18 | 0.2246/0.2242 | 0.2449/0.2463 |
| 3,1,2,1,1 N | 13 | 0.2487 | 0.2667 |
| 3,1,3,0,1 N | 13 | 0.2389 | 0.2522 |
| 3,0,3,1,1 N | 13 | 0.2621 | 0.2789 |
| **3,1,3,1,2 N** | **15** | **0.2259** | **0.2426** |
| 3,1,3,1,3 N | 16 | 0.2245 | 0.2421 |
| 3,2,3,1,2 N | 16 | 0.2239 | 0.2413 |
| 4,1,3,1,2 N | 16 | 0.2253 | 0.2427 |
| 3,1,4,1,2 N | 16 | 0.2256 | 0.2440 |

Results:
- Cubic means are needed: dropping to quadratic costs about 0.02.
- Both log-sd functions need a slope.
- Higher degrees, Student-t components and freer weights do not help beyond noise.

## Choice
Spec (3,1,3,1,2): two Gaussians with cubic means, log-linear scales and a logistic-quadratic weight.
It has 15 fitted constants (19 literals in the file, including 2.0/1.0 in z and 1.0 - w).
Its CV score is within 0.0015 of the best variant, which has one more parameter, so I chose it
for parsimony. The full-data training NLL is 0.2259.

Because I picked this model from about 16 candidates, its CV score is slightly optimistic, so I add a
small allowance to the estimate.

expected_nll: 0.245
