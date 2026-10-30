# Notes

## Validation
10-fold CV repeated 2-3 times (fixed seeds), with the whole fitting procedure (including
standardisation and optimisation) redone inside each fold. Metric: mean NLL of y.

## What I tried (CV NLL)
- Marginal normal: 1.058
- Ridge on all 25 covariates + Normal: 0.626; + Student-t: 0.594
- Forward-selected linear mean + t (selection is optimistic): ~0.588
- Linear mean jointly fitted with a 2-component normal-mixture error: 0.50-0.53;
  under the mixture, x24 alone was best (0.464). Extra covariates made it worse.
- Key finding: y <= x24 for nearly all rows (only 7 of 804 have y > x24, by at most 0.02), and
  many rows sit right at x24. So y = x24 - D with D >= 0 and right-skewed. The scale of D
  shrinks as gap = x24 - max(x1..x23) grows (corr of log D with gap is -0.44).
- Mixture of strongly left-skewed SkewNormals at loc x24 + offset, log-scale linear in gap:
  K=1: 0.401 (no gap) / 0.386 (gap); K=2: 0.361 / 0.325; K=3: 0.342 / 0.292.
  On the full data, K=3 collapses to two live components (third weight ~1e-30, 12 restarts),
  so the K=2 fold fits that scored 0.325 were probably stuck in worse local optima. The final
  model is that 2-component solution.

## Final program
g = clip(x24 - max(x1..x23), -0.224, 1.17) (the clip keeps scales bounded anywhere in the box);
y ~ 0.861 * SkewNormal(x24 + 0.0034, exp(-0.466 - 3.754 g), -56.8)
  + 0.139 * SkewNormal(x24 - 0.402, exp(-0.141 - 0.543 g), -70.9).
That is 12 constants (286 AST nodes), counting the clip bounds. In-sample NLL is 0.256. The CV of the
procedure that lands on this solution is 0.292, so my estimate for new data is ~0.30.
Time ran out before I could try covariate-dependent weights or offsets, or other subsets of
columns for the max.

expected_nll: 0.30
