# Notes

## Data findings
- x1 ≈ 1 − x2 (sd 0.0015), so it adds nothing; it is unused.
- x0 sits on a grid of 3.0, 3.1, …, 6.0 (noise sd 0.009), and x2 on a grid of 0, 0.01, …, 0.35 (noise sd 0.001).
- x3 is in effect a noisy linear reading of y: y ≈ 1.18 + 0.001·x3, with residual sd 0.0028. The exception is 56 of 448 rows (12.5%), where x3 ≈ 0 (|x3| < 5), i.e. it is "missing". Observed x3 values are all ≥ 37, so the gate `x3 > 20` is safe.
- y as a function of (x0, x2) is a 2-D staircase:
  - Along x0 it is a sawtooth with period 0.5, breaking at 3.05, 3.55, …. Within a block y falls by about 0.015 per grid step, and the blocks rise from one to the next.
  - Along x2, y is flat for x2 < 0.1. Above that it steps up every 0.05 (at 0.10, 0.15, …), with step sizes that shrink and depend on the x0 block, and falls slightly within each step.
- The sawtooth features use arctan2(sin, cos) to get the in-block position f and the block index b = x/P − f.

## Final program
- **If x3 > 20:** Normal(a0 + a1·x3 + a2·g(x), 0.00276).
- **Otherwise:** StudentT(df 4, g(x), s), where g is linear in 16 features built from b0, f0, s2 = max(b2 − 1, 0), g2 (the in-block x2 position, set to 0 below x2 = 0.1), their interactions, s2², sqrt(s2), b0·s2², and an indicator for the partial block at x0 ≈ 3.0.
- g is fitted by OLS on all rows, since y relates to x0 and x2 the same way whether or not x3 is missing.
- 40 literals, 364 AST nodes.

## Validation
I used 10-fold CV repeated 5 times, refitting the whole pipeline in every fold.

Progression of the missing-x3 branch (CV NLL on all rows):

| Model for the missing-x3 branch | CV NLL |
|---|---|
| Linear in x0, x1, x2 | −1.43 |
| Quadratic | −1.54 |
| Sawtooth blocks + interactions | −2.98 |
| + extra terms | −3.14 |
| Student-t df 4 (instead of Normal) | −3.21 |

Final CV NLL of the full model:

| Rows | CV NLL |
|---|---|
| All | −4.304 (sd over repeats 0.003) |
| Observed x3 | −4.470 |
| Missing x3 | −3.14 |

The training NLL is −4.324. The score also depends on what fraction of the held-out rows have x3 missing. I allow a little for optimism and for edge cells (x2 ≈ 0.35, x0 ≈ 3.0) that the data covers poorly.

expected_nll: -4.28
