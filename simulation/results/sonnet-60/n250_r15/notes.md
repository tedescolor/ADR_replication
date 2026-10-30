# Notes
Data: 1 covariate; y shows a smooth curve for x<0.6 that splits into two branches (one rising to ~2.5, one falling to ~-1) for x>0.6; noise is larger near x=0.
Model: 2-component Gaussian mixture. Component 1 mean is a cubic in x, component 2 mean is a quadratic. Each log-sd is linear in x. The weight of component 2 is sigmoid(linear in x). 14 constants in total, fitted by penalised ML (BFGS, multi-start).
Validation: 5-fold CV on the full pipeline. Mean NLL by (deg1, deg2): (2,1) 0.505, (3,1) 0.389, (4,1) 0.377, (5,1) 0.354, (3,2) 0.331, (4,2) 0.330, (5,2) 0.329, (3,3) 0.327.
I chose (3,2): extra degrees gain under 0.005 and add overfitting risk. Training NLL is 0.247, so the CV figure is the honest one.
Not tried: heavy-tailed components, or a weight that is non-linear in x.

expected_nll: 0.34
