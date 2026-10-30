# Notes

## Data
1000 rows, x0 in (0,1), y in about (-1.5, 2.7). Scatter: a single noisy trunk for x < ~0.5
(sd ~0.35 at low x, ~0.22 near 0.45). Past ~0.5 it splits into two branches of roughly equal
weight: an upper one rising almost linearly to ~2.5 at x=1, and a lower one that peaks
around x~0.65 and drops to ~-1 at x=1. Both branches are tight (sd ~0.1-0.2).

## Validation
5-fold CV repeated twice (10 fits per candidate; different seeds and multiple starts for
each fit), with held-out NLL pooled over all rows. Scored on the original y scale.

## Candidates (CV mean NLL)
Two-Gaussian mixture, polynomial means in t = x - 0.5, log-linear/poly sds, logistic-poly weight
(degrees mean, logsd, weight):
- 3,1,1: train 0.1708, CV **0.1847**  <- chosen
- 3,1,2: 0.1862; 3,2,1: 0.1878; 4,1,1: 0.1869; 4,2,1: 0.1913; 3,0,1: 0.272

"Structured" mixtures where the second branch only switches on past a learned split point
(sigmoid gate, about 0.5), with poly means, poly log-sds, an optional jump in the trunk's
log-sd at the split, and Normal or Student-t components: CV 0.187-0.200. These were no better,
because in the simple mixture the overlapping components at low x also capture the
non-Gaussian (platykurtic) trunk noise.

## Final
Mixture of 2 Normals: cubic means, log-linear sds, logistic-linear weight; 14 fitted
constants plus the centring 0.5 and the `1 - w`. I chose the simplest model with the best CV.
The CV differences among the top candidates are about 0.002, which is within noise.

expected_nll: 0.185
