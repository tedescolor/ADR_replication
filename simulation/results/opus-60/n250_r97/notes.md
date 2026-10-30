# Notes

Data: 250 rows, one covariate x0 in (0,1). Scatter shows a single noisy curve for x<0.5
(dips to about -0.7 near x=0.3, sd about 0.3), which splits into two tight branches above x≈0.55:
an upper branch rising almost linearly to about 2.6 at x=1 and a lower branch falling to about -0.9.
Branch noise shrinks as x grows (sd about 0.1 at x=1).

Model family: 2-component Gaussian mixture with polynomial means in t=2x-1,
log-sd linear in t, and mixing logit quadratic in t; fit by maximum likelihood (BFGS, multiple starts).

Validation: 10-fold CV (shuffled), refit on each training fold, mean held-out NLL:
- means deg 3, logsd deg 1, logit deg 2 (15 params): train 0.071, CV 0.141
- means deg 4, logsd deg 1, logit deg 2: CV 0.163
- means deg 4, logsd deg 2, logit deg 2: CV 0.206
- means deg 5, logsd deg 2, logit deg 2: CV 0.236
More flexible variants overfit (or hit poorer optima), so I picked the smallest one.

Structured variants (same 10-fold x 2 repeats split): shared centre curve c(x) plus a hinge split
h(x)=0.5*u*(k1+k2*u), u=max(x-x0,0), branches c±h, a shared sd exp(a+b x), and a constant weight:
- A two cubics, shared linear logsd, constant weight: CV 0.189
- B two quartics, separate logsd, linear logit: CV 0.150
- C quartic centre + hinge split: CV 0.135
- D C with quadratic logsd: CV 0.139
- E quintic centre + hinge split (12 params): train 0.074, CV 0.127 (se about 0.05)
- F C with a separate logsd per branch: CV 0.140

Final: model E, refit on all data (12 fitted parameters; hinge at x≈0.37, weights 0.39/0.61).
It had the best CV score and is simpler than the unstructured mixtures. expected_nll is the CV
value plus a small margin for selection optimism.

expected_nll: 0.14
