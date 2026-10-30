# Notes
Data: 1 covariate on (0,1); mean is nonlinear (dip near x=0.2, rising after 0.5), spread grows strongly with x (sd ~0.3 -> ~1.4).
Candidates: Normal / Student-t, polynomial mean (deg 3-6), polynomial log-sigma (deg 0-4), fit by ML.
Validation: repeated (3x) 10-fold CV, mean held-out NLL. Heteroscedasticity was the key (const sigma ~1.0, linear ~0.73, quadratic ~0.585).
Higher degrees / t tails gave no real gain (all ~0.583-0.59). Final: quartic mean, quadratic log-sigma, StudentT (df fitted ~3000, i.e. effectively Normal); 9 constants + 0.5 centering.
Train NLL 0.556; CV NLL ~0.583 (slightly optimistic from model selection).
expected_nll: 0.60
