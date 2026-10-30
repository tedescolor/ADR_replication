# Notes

## Data
1 covariate x0 in [0,1], continuous y, n=1000. The scatter shows a single noisy branch for
x < ~0.6 (sd ~0.3, U-shaped mean) that splits at x ~0.615 into two branches: an upper one
rising to ~2.5 and a lower one falling to ~-0.9 at x=1. The noise shrinks with x (~0.1 at x=1).
Standardised within-branch residuals look Gaussian (skew ~0, excess kurtosis ~0.2 or less),
so Gaussian components are enough.

## Model
A two-component Gaussian mixture, with t = 2*x - 1:
- upper branch: mean is a degree-6 polynomial in t; log-sd is quadratic in t;
- lower branch: mean is a cubic polynomial in t; sd = constant ratio * upper sd (shared noise shape);
- lower-branch weight = sigmoid(a) * sigmoid(k*(x - x_s)), which is ~0 before the split and ~0.56 after it.
The program has 22 fitted constants (about 25 literals in all). It is fitted by maximum likelihood (L-BFGS, 5 random restarts).

## Validation
5-fold CV, repeated with 2 different shuffles (10 held-out folds). Mean held-out NLL:
- generic 2-Gaussian mixture, polynomial means/log-sds and a polynomial logit weight: 0.172–0.173
- split-weight model, separate sd polynomials for each branch (various degrees): 0.169–0.173
- split-weight model, shared sd shape (degrees (5–7, 3–4, 2)): 0.1635–0.1646; with linear log-sd: 0.168
I chose (upper deg 6, lower deg 3, log-sd deg 2, shared): CV 0.1635, train NLL 0.1415.
Models near this one all score about the same, so the choice is not sensitive to the exact degrees.

## Estimate
CV is 0.1635. The final model is refit on all 1000 rows, which gives a slight gain, and
selecting the best of several models gives a slight optimistic bias. The two roughly cancel.

expected_nll: 0.164
