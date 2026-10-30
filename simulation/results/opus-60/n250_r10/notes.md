# Notes

## Data
250 rows, x0 in (0,1). Scatter shows two regimes:
- a wavy "main" curve over the whole range (min about -0.9 at x≈0.22, max about 0.5 at x≈0.6,
  then falling again), with noise sd that shrinks as x grows (about 0.35 down to 0.16);
- for x > ~0.69, about half the points lie on a second, nearly noise-free straight line,
  y ≈ -2.08 + 4.67 x (sd ≈ 0.097). The smallest x on that line is 0.694.

## Model
A two-component mixture:
- main: StudentT(df≈14.8, loc = a + b x + c sin(w x + φ), scale = exp(α + β x));
- upper branch: Normal(-2.08 + 4.67 x, 0.0965);
- weight of the upper branch = sigmoid(-0.059) (≈0.485) * sigmoid((x - 0.69)/0.002), so it is a sharp step.
The program has 15 constants.

## Validation
5-fold CV, repeated 3 times (each point's held-out NLL averaged over the repeats). Refits were
warm-started from the full-data fit. Candidates (CV NLL):
- sin mean, Normal, sd linear in log, step at 0.69/0.002: 0.1963 (train 0.123)
- same, but with a Student-t main component: 0.1965 (train 0.120)
- sin + quadratic term: 0.202 for both the Normal and t versions (worse)
- softer or earlier steps (0.68/0.005, 0.66/0.01): 0.198-0.210 (worse)
- constant noise sd (an earlier, slower run): about 0.26 (worse)
The largest held-out losses come from a few isolated main-curve outliers: (0.30,-1.68),
(0.60,1.03), (0.62,1.03), (0.05,0.68).
I chose the t version: its CV score is the same as the Normal one, and its heavier tails cost
less on those outliers.

Estimate: the CV value is ≈0.197. I add a small allowance for model selection and for the
uncertain switch location.

expected_nll: 0.20
