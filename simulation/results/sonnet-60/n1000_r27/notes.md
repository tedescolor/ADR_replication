# Notes
Data: 1 covariate x0 in [0,1]. Single Normal with polynomial mean/log-sd: 5-fold CV NLL ~0.626
(StudentT/SkewNormal no gain). Histograms at high x showed two well-separated branches (bimodal), so
I moved to a 2-component Normal mixture: cubic means, log-sd linear in x, mixture logit quadratic in x.
Validation: 5-fold CV (shuffled), refit each fold, several restarts. CV NLL by (mean deg, logsd deg, logit deg):
(2,1,1) 0.48, (3,1,1) 0.204, (3,1,2) 0.201 (chosen), (3,2,2) 0.205, (4,*,*) ~0.214+, (3,1,0) 0.207, (3,0,*) ~0.30.
Final: (3,1,2) fit on all data (train NLL 0.184), 15 constants + 1 for `1 - w`. Passes the contract check.
Expected NLL is the CV estimate, slightly padded.
expected_nll: 0.205
