# Notes
Data: y vs x0 shows one noisy branch for x<0.5 that splits into two narrow branches (rising and falling) for x>0.5.
Model: 2-component Normal mixture; branch-1 mean quartic in x, branch-2 mean quadratic, mixture weight sigmoid(quadratic), log-sd linear per branch. 17 fitted constants (+ a few structural), fit by ML (BFGS, multi-start).
Validation: 5-fold CV (shuffled) over polynomial orders; chosen (5,3,3,2,2) coefficient counts: CV NLL 0.223 (train 0.206). Higher orders gave no gain; lower orders were worse (0.24-0.27).
Final fit on all data, contract check passes.
expected_nll: 0.23
