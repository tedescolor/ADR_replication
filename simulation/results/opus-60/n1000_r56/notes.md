# Notes

## Data
1000 rows, x0 in (0,1), y in about (-1.8, 2.6). Looking at a scatter plot, there's one noisy band for
x < ~0.57 (U-shaped: about -0.3 at x=0, -0.9 at x≈0.25, then rising). At x≈0.57 it splits
into two branches: an upper branch that keeps rising to about 2.5 at x=1, and a lower branch that
stays flat near 0.1 and then falls to about -1 at x=1. Each branch has roughly equal weight, and
the noise looks Gaussian within a branch. I looked for a jump near x=0.4 and didn't find one.

## Validation
5-fold CV (fixed random permutation), fitting by maximum likelihood with BFGS and a few
restarts. In CV, each fold also starts from the full-data fit, which adds only a small
selection effect through the choice of local optimum. Folds were pooled for each candidate.

## Candidates (CV NLL / train NLL)
- 2-component Normal mixture, polynomial means, polynomial log-sd, polynomial logit weight:
  - degrees (3,3,1,1,1): 0.228 / 0.216
  - (4,4,2,2,2): 0.226
  - (5,5,2,2,3): 0.2175 (on a second CV run: 0.222)
  - (6,6,2,2,3): 0.236. This overfits and is unstable.
- Structured gate: lower-branch weight = sigmoid(a)*sigmoid(k*(x-0.5-c)). Mean
  degrees (m_upper, m_lower) and log-sd degrees (s_upper, s_lower):
  - (5,3,2,1): **0.2150** / 0.198, with 18 parameters
  - (6,3,2,1): 0.2146
  - (7,3,2,1): 0.2151
  - (6,3,3,1): 0.2139
  - (5,3,2,0): 0.2207
  - (5,2,2,1): 0.2172

## Final program
I chose the structured gate with degrees (5,3,2,1). It's within noise of the best candidate
and has fewer parameters. The fitted gate turned into a hard step at x≈0.566: for x below it,
the program uses a single Normal (the upper branch, quintic mean, quadratic log-sd). For x
above it, it's a mixture with weights 0.51/0.49, where the lower branch has a cubic mean and
a linear log-sd. The step slope is written as 1e6. The program has 20 literals and passes the
contract check. Its NLL on the training data is 0.198.

The estimate below is the CV value plus a small allowance for having chosen among about 10
candidates.

expected_nll: 0.22
