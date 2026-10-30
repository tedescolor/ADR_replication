# Notes

## Data
There is one covariate x0 in [0,1]. The scatter shows two smooth curves. They overlap and cross near x≈0.5 and split apart for x>0.55. One branch rises linearly to about 2.5; the other bends down to about -0.9. The noise shrinks as x grows, from a scale of about 0.3 near x=0 to about 0.1 near x=1. Within bins, kurtosis at large x is about -2, which is bimodal.

## Model family
The model is a 2-component mixture. Each component mean is a polynomial in x. Each log-scale is a polynomial in x. The mixture logit is a polynomial in x. All parameters were fitted jointly by maximum likelihood (BFGS, 6 random restarts).

## Validation
I used 5-fold CV (one fixed permutation) and compared configurations by mean held-out NLL. The degrees are written as (mean1, mean2, logsd1, logsd2, logit).
- (3,4,2,2,2) N: 0.2298
- (3,4,1,1,0) N: 0.2244
- (3,4,2,2,0) N: 0.2234
- (3,4,2,2,1) N: 0.2258
- (4,4,2,2,2) N: 0.2223
- (4,4,2,2,0) N: 0.2190
- (4,4,3,3,0) N: 0.2192
- (4,4,1,1,0) N: 0.2243
- (5,5,2,2,0) N: 0.2211
- (4,5,2,2,0) N: 0.2225
- (4,4,3,2,0) N: 0.2190
- (4,4,2,2,0) Student-t: 0.2185, with nu1≈22 and nu2≈2000 (so component 2 is effectively Normal)

Letting the weight depend on x made CV worse, so the weight is constant (≈0.53). The folds vary a lot: 0.11 to 0.41. Fold 3 holds several mild outliers.

## Final model
- Quartic means.
- Quadratic log-scales.
- Constant weight.
- Component 1 is Student-t with nu=21.7; component 2 is Normal.
- 18 constants.
- Training NLL: 0.1995. 5-fold CV NLL: 0.2185.
- The fold-to-fold spread is large, so I round the estimate up slightly.

expected_nll: 0.22
