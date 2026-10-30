# Notes

Data: 1000 rows, x0 in [0,1]. Scatter shows a single noisy band for x<0.55 (sd ~0.4 near 0,
shrinking to ~0.2 near 0.5) that splits into two sharp branches for x>0.55: one rising roughly
linearly to y~2.5, one falling to y~-1, each with sd ~0.1 and roughly equal weight.

Model family: 2-component Gaussian mixture on the original y scale, with t = x0 - 0.5,
polynomial means, log-sd polynomials and a logistic weight. Spec (m1,s1,m2,s2,w) = number
of coefficients per polynomial. Fitted by direct ML (BFGS, multiple starts).

Validation: 5-fold CV repeated over 2 random partitions (mean held-out NLL per row).
- (5,3,5,3,3) 0.2097   (5,3,5,3,2) 0.2089   (5,3,5,3,1) 0.2130
- (5,2,5,3,3) 0.2185   (5,3,5,2,3) 0.2107   (5,4,5,4,3) 0.2190
- (4,3,5,3,3) 0.2096   (5,3,4,3,3) 0.2164   (4,3,5,3,2) 0.2102
- (4,2,4,2,1) 0.2211 (single split), (6,4,6,4,4) 0.2137 (single split)
- Student-t components: fitted df -> ~1000+, CV 0.2089 (no gain) -> Gaussian kept.

Final: (5,3,5,3,2) refit on all data: quartic means, quadratic log-sds, linear logit
weight; 18 constants (+1 for the 0.5 centring and 1 in 1-w). Train NLL 0.191, CV 0.209.
Expected NLL on new data ~ CV value (slightly optimistic due to spec selection).

expected_nll: 0.21
