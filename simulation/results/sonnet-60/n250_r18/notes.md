# Notes
Data: y vs x0 is bimodal at high x (a rising branch and a falling branch), heteroscedastic.
Model: 2-component normal mixture; component means are polynomials in x0 (degree 4), log-sd linear in x0, mixing weight = sigmoid(quadratic). 17 constants.
Validation: 5-fold CV (shuffled), refit with multi-start BFGS per fold. CV NLL (dm,ds,dw):
(2,0,1) 0.597; (3,0,1) 0.436; (3,1,1) 0.260; (4,1,2) 0.250 (chosen); (4,1,1) 0.273; (4,2,*) ~0.32; (5,1,2) 0.280.
Train NLL of final fit 0.154 (optimistic). Expected adds some margin for model selection over folds.
expected_nll: 0.27
