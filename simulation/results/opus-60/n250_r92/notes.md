# Notes

## Data
250 rows, x0 in [0,1]. The scatter plot shows two curves. One dips to about -0.9 near x=0.2 and
climbs to about 2.4 at x=1. The other is flatter, peaks near 0.1 around x=0.7 and drops to about
-1 at x=1. The curves cross near x~0.5, and they separate clearly for x>0.6. Noise shrinks as x
grows: sd is about 0.3 at x=0 and about 0.1 at x=1.

## Models tried
All are 2-component Gaussian mixtures fitted by maximum likelihood (BFGS, multi-start).
Validation: 5-fold CV repeated 2 times with different shuffles, refitting on each training
fold. Numbers are mean held-out NLL; the SE is about 0.054, but the differences between models
are paired, so they are much more precise than that.

- cubic means, linear log-sd per component, logit-linear weight: 0.231
- cubic means, polynomial degrees 4/5 for one mean: 0.24-0.26 (worse)
- quadratic/cubic combinations with lower degree: 0.26-0.59 (worse)
- constant sd: 0.27-0.31 (worse)
- per-component linear log-sd, constant weight: 0.216
- shared linear log-sd, constant weight: 0.2006
- shared log-sd plus x-dependent weight: 0.207
- quartic for one mean: 0.203
- shared quadratic log-sd: 0.202
- linear + sinusoid means: these collapsed into the cubic solution and scored the same, about 0.20
- **shared linear log-sd, fixed equal weights 0.5/0.5: 0.1926** (best; 10 fitted constants)

## Final model
y | x ~ 0.5 N(m1(x), s(x)) + 0.5 N(m2(x), s(x)), where m1 and m2 are cubic polynomials and
s(x) = exp(a + b x). Training NLL is 0.143; CV NLL is 0.193. I picked this model out of about
12 candidates, which biases the CV score slightly downward, so I add a small amount for that.

expected_nll: 0.20
