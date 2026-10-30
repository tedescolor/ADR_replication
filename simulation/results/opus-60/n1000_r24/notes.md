# Notes

## Data
1000 rows, x0 in [0,1], y in [-1.53, 2.65]. Scatter plot shows:
- a smooth "lower" curve over the whole x range (~-0.2 at 0, min ~-0.7 at x~0.25,
  ~+0.2 at x~0.6, down to ~-1 at x=1), noise sd decreasing from ~0.37 to ~0.10;
- a second, nearly linear "upper" branch that splits off at x~0.45-0.55 and rises to ~2.5 at
  x=1, carrying ~40-45% of the mass for x>0.55, sd ~0.15-0.2.

## Model family
2-component mixture, polynomials in t = x-0.5:
- lower: Normal(poly6(t), exp(poly2(t)))
- upper: StudentT(df, poly2(t), exp(poly1(t)))
- upper weight = sigmoid(c) * sigmoid(k (x - x0)) (plateau times onset; the fitted onset is
  essentially a step, k is capped at 2000 in the program).
Fitted by maximum likelihood (BFGS + Nelder-Mead polish), initialised from a hand split of the
two branches. A first attempt with a plain logistic weight (no plateau) got stuck in a bad
optimum (CV 0.64).

## Validation
5-fold CV repeated over 2 seeds (10 held-out folds), refitting each fold from the full-data
optimum. CV mean NLL (lower = better):

| degrees: lower mean / lower log-sd / upper mean / upper log-sd, family | train | CV |
|---|---|---|
| 5/1/1/0 Normal | 0.231 | 0.243 |
| 6/1/1/0 Normal | 0.229 | 0.242 |
| 6/2/1/1 Normal | 0.218 | 0.234 |
| 6/3/1/1 Normal | 0.218 | 0.235 |
| 8/2/1/1 Normal | 0.211 | 0.235 |
| 6/2/2/1 Normal | 0.211 | 0.232 |
| 6/2/1/1 upper-t | 0.215 | 0.232 |
| **6/2/2/1 upper-t (final)** | 0.209 | **0.2255** |
| 6/2/2/2 upper-t | 0.207 | 0.226 |
| 6/2/2/1 both t | 0.209 | 0.226 (lower df -> inf) |
| 7/2/2/1 upper-t | 0.207 | 0.225 |

(degrees above are polynomial degrees.) Lower-component t gave nothing; extra degrees beyond
the final configuration gave <=0.0005 improvements, so I kept the simpler one.

## Final program
19 fitted constants + the 0.5 centring + the 2000 onset slope = 21 literals, 197 AST nodes.
Train NLL 0.209, CV NLL 0.2255. Some optimism remains from model selection over ~12 configs,
so I add a small margin.

expected_nll: 0.23
