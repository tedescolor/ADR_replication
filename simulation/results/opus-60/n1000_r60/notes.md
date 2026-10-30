# Notes

**Data.** 1000 rows, x0 in (0,1). The scatter shows two branches. A lower, wavy curve runs across all x
(0 at x=0, about -0.7 at x≈0.25, about +0.2 at x≈0.65, about -1 at x=1); its noise sd shrinks from about 0.4 to 0.1.
For x > ~0.45 an upper branch rises roughly linearly from about 0.3 to 2.5, with sd about 0.1–0.2.
For x>0.6, roughly half the points sit on each branch.

**Model family.** A 2-component Gaussian mixture. Every parameter is a polynomial in t = 2*x0 - 1:
the two means, the two log-sds and the logit of the upper-branch weight. Fitting is by direct maximum likelihood
(BFGS, Chebyshev basis for conditioning). Initialisation comes from a hard split of the branches (y > 5x - 2.75, x > 0.45).
A first attempt with raw monomials and naive inits converged badly; the Chebyshev basis plus the branch split fixed this.

**Validation.** 5-fold CV repeated twice (10 folds), refitting from scratch in each fold. Mean held-out NLL (SE about 0.013):
- degrees (m1,m2,s1,s2,w) = (5,2,1,1,1): 0.230; (8,2,2,1,1): 0.225
- (7,3,2,2,2): 0.194; (7,3,3,2,2): 0.194; (9,3,3,2,3): 0.193
- Student-t components instead of Gaussian: 0.194 (no gain; the tails are Gaussian)
- capped weight c*sigmoid(a+bx): 0.201 (worse)
- (6,3,2,1,3): 0.1914 <- chosen; (6,3,2,1,4): 0.189 (within noise, more constants)
Once the scale and weight functions were flexible enough (degree ≥ 2), every variant fell within about 1 SE of the others.
I picked a mid-sized one with 20 constants.

**Final.** Refit on all data: train NLL 0.173, CV 0.191. Fitting on the full 1000 rows rather than 800 should help a little,
but the model selection adds a small optimism. My estimate is about 0.19.

expected_nll: 0.19
