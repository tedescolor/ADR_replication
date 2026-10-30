# Notes

## Data
250 rows, x0 in [0,1]. Scatter shows a single noisy sinusoidal curve for x < ~0.5;
above x ~ 0.5 the data split into two branches: the sinusoid continues (lower branch,
peak ~0.2 at x~0.65, back to ~-0.8 at x=1) and a second, rising branch appears
(~0.5 at x=0.55 up to ~2.3 at x=0.95), holding about 45% of the points beyond x=0.5.
The lower-branch noise shrinks with x (sd ~0.36 near x=0, ~0.16 near x=1); its
standardized residuals are slightly light-tailed, so Normal components are adequate.

## Models compared (2 x 10-fold CV, refitting everything inside each fold; mean held-out NLL)
All are 2-component Normal mixtures. Lower mean = a sin(b x + c) + e + g x, lower log-sd linear in x.
- base: upper mean linear, upper log-sd linear, weight = sigmoid(w0 + w1 x): 0.325
- q (upper quadratic): 0.331; rel (upper = lower + slope*relu(x-x0)): 0.337
- sq (quadratic lower log-sd): 0.336; cs (constant upper sd): 0.318
- step: weight = sigmoid(w) * sigmoid(60 (x - x0)) (branch switches on at x0): 0.305
- stepk (learned sharpness, went to a hard step): 0.308
- stepcs: 0.309; stepqcs: 0.311; relstep: 0.308
- **stepq** (step weight + quadratic upper mean + linear upper log-sd): **0.3006** (train 0.233)
- stepqk: 0.301
Earlier exploratory polynomial-mean mixtures were slower and not pursued.

## Final choice
stepq: 14 fitted parameters (15 literals incl. the fixed step sharpness 60), refit on all data.
The switch point fitted at x0 = 0.501, upper-branch weight sigmoid(-0.145) = 0.46.
The step-type weight was consistently ~0.02 better than a smooth logistic weight;
differences among the step variants are within CV noise, so stepq was picked as the best CV.
Expected NLL on new data: the CV estimate ~0.30, nudged up slightly for selection among
~10 variants.

expected_nll: 0.31
