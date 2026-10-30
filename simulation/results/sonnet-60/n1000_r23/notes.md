# Notes
Data: y is bimodal in x0 — a lower sinusoidal-looking branch and an upper, roughly linear branch that appears for x0 > ~0.5. Noise shrinks with x0 within each branch.
Model: 2-component Normal mixture. Lower mean = a + b*sin(w*x+phi), upper mean linear in x, log-sd linear in x for each branch, and the upper weight is sigmoid(linear in x). 13 constants in total.
Fit by ML (L-BFGS, multi-start). Validation: 5-fold CV, with the model refit on each training fold.
- Upper mean quadratic: CV 0.277 (training 0.240). Not better, so rejected.
- Upper mean linear (chosen): CV 0.272 (training 0.257).
I tried no other model families and did no further tuning. The remaining gap between training and CV NLL is moderate.
expected_nll: 0.28
