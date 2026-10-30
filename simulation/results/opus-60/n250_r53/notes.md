# Notes

## Data
250 rows, x0 in (0,1). Scatter shows one noisy curve (dip to ~-0.9 at x~0.25) up to x~0.56,
where y bifurcates: an upper branch continuing the same smooth curve (to ~2.6 at x=1) and a
lower branch bending down (to ~-0.9 at x=1). About half (0.47) of points with x>0.56 are on the lower branch.

## Model family
Two-component Normal mixture:
- main branch mean: cubic in x; lower-branch mean: quadratic in x;
- lower-branch weight w = c * sigmoid(k (x - t)) (fitted k is huge, i.e. a hard switch at t=0.562);
- log sd polynomial in x (separate per branch or shared).
Fit by maximum likelihood (BFGS, several restarts).

## Validation
5-fold CV repeated with 2 seeds (10-fold x2 for the first batch), refitting all parameters in each fold.
Mean held-out NLL:
- separate log-linear sds, fixed weight 0.5: 0.243 (higher-degree mean/sd polys: 0.246-0.262)
- separate log-linear sds, fitted weight: 0.236
- shared log-quadratic sd: 0.229; shared log-cubic: 0.234; shared log-quartic: 0.251
- quartic main mean + shared log-quadratic sd: 0.238
- **shared log-linear sd (final, 12 fitted constants): 0.227** (train 0.196)

The simplest variant validated best, so I chose it. The final model refits it on all the data.

## Estimate
The CV includes some optimism, because I picked the best of about 12 configurations and the
fold fits were started from the full-data fit. I add a small margin.

Process note: for a few minutes three fitting processes overlapped, because an earlier
background job was not killed when I intended. This was above the 2-process limit.

expected_nll: 0.24
