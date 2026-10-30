# Notes

## Data
1000 rows, x0 in [0,1], y in [-1.6, 2.7]. Scatter shows a lower, wavy branch over the whole
range (about -0.2 at x=0, min about -0.9 near x=0.22, peak about 0.2 near x=0.6, about -1 at x=1),
plus an upper branch that splits off near x=0.5 and rises almost linearly to about 2.5 at x=1.
Noise shrinks as x grows: lower-branch sd goes from about 0.33 to 0.1, upper branch about 0.1.
Within each component the standardized residuals look Gaussian (skew about 0.1, excess kurtosis about 0).

## Model family
Two-component Gaussian mixture. Each component mean and log-sd is a Legendre polynomial in
u = 2x-1, and the mixing weight is sigmoid(polynomial in u). It is fitted by direct MLE with
analytic gradients (L-BFGS-B) and several random restarts. The likelihood has strong local optima:
my first fits got stuck at train NLL 0.588 versus 0.22 at a good optimum. That was caused by a bad
init for the upper-branch mean, and I fixed it.

## Validation
5-fold CV repeated 3 times (15 folds, the same splits for every config), refitting with restarts
inside each fold. Mean held-out NLL (SE about 0.015 across folds, but the configs are compared
paired):
- deg(m1,m2,l1,l2,w) = (6,2,2,1,2): 0.2496
- (6,3,2,1,2): 0.2494; (6,3,2,2,2): 0.2454; (6,3,3,1,2): 0.2469
- (6,3,2,1,1): **0.2426**; (5,3,2,1,1): 0.2434; (6,3,2,2,1): 0.2437
- fewer sd terms or a lower-degree upper mean were worse (0.249 to 0.258)

## Final
(6,3,2,1,1): 18 fitted coefficients written as Horner polynomials, plus the 2.0 and 1.0 of the
u transform and the 1 in 1 - w (21 literals). Refitted on all data, train NLL 0.2222.
The CV estimate includes the refitting and optimizer variability. Full-data refit should be about
as good or a bit better, so I take the CV value with a small allowance for selection.

expected_nll: 0.245
