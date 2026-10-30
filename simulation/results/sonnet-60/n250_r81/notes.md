# Notes
Data: 1 covariate x0 in (0,1). The mean of y is nonlinear in x0 and spread grows with x0. For x0 > ~0.65 y is bimodal: an upper branch rising to ~2.4 and a lower branch falling to ~-1.

Candidates: 2-component Normal mixture with polynomial means (degrees d1,d2), log-linear sds, and a logistic mixing weight in x0.
Validation: 5-fold CV (one random split, seed 1), 8 random restarts per fold, mean held-out NLL:
- (2,1): 0.563
- (3,1): 0.342
- (3,2): 0.321
- (4,1): 0.338
- (4,2): 0.287 (chosen)
- (5,2): 0.315

Final: (4,2), refit on all data with 40 restarts (train NLL 0.221). The structure is Mixture of 2 Normals with 14 constants plus the 1 in `1 - w`. The 5-fold CV figure is mildly optimistic because I chose the degrees by CV on the same data.

expected_nll: 0.31
