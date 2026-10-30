# Notes

**Data.** x0 is roughly uniform on [0,1]. A scatter plot shows two branches. The lower branch is a wavy curve that exists for every x: its value is about -0.2 at x=0, -0.8 at x=0.25, +0.1 at x=0.7 and -1 at x=1. Its noise shrinks with x (sd falls from about 0.38 to 0.1). The upper branch starts at x≈0.53, where it splits off the lower curve, and rises roughly linearly to about 2.5 at x=1. About 46% of the points with x>0.53 belong to it.

**Model family.** A 2-component Normal mixture, fitted by maximum likelihood with BFGS from several starting points:
- lower mean: polynomial in (x-0.5);
- upper mean: polynomial in (x-0.5);
- scales: exp(linear), exp(quadratic) or linear in x;
- upper weight: a constant times a sigmoid step in x.

**Validation.** 5-fold CV mean NLL, with each fold refitted from the full-data optimum. Shuffles used 1–3 seeds.

| variant | params | train | CV |
|---|---|---|---|
| poly logistic weight (deg 6/2) | 17 | 0.223 | 0.241 |
| step weight, lower deg 6, upper deg 2, exp-linear sd | 17 | 0.198 | 0.214 |
| lower deg 5 / sinusoid lower | 15-16 | 0.207-0.213 | 0.222-0.230 |
| lower deg 7 / 8 | 18-19 | ~0.198 | 0.213-0.215 |
| upper deg 3 | 18 | 0.195 | 0.2116 |
| upper deg 3 + linear sd | 18 | 0.194 | 0.2115 |
| exp-quadratic sd | 20 | 0.193 | 0.214 |

The step weight was the main gain. The differences among the last few variants are within CV noise (about 0.005).

**Final.** Lower mean: degree-6 polynomial. Upper mean: cubic. Both sds linear in x, floored at 0.02. Weight: sigmoid(c)·sigmoid((x-0.5257)/0.005). The step width is fixed rather than fitted, because the fitted value was effectively a hard step, which would be fragile. 17 fitted parameters and 22 literals.
- Train NLL: 0.195
- CV NLL with the final specification (5-fold × 3 seeds): 0.217

expected_nll: 0.216
