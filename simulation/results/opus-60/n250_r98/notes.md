# Notes

## Data
250 rows, x0 in (0,1), y in about (-1.5, 2.7). For x < ~0.6, y has a single noisy branch whose mean
looks like a sinusoid (about -0.2 at x=0, a trough of -0.85 near x=0.25, about 0.1 near x=0.55),
with sd of about 0.35. For x > ~0.6, y is clearly bimodal: an upper branch rises roughly linearly
to about 2.5 at x=1, and a lower branch falls to about -0.85. Within each branch the noise is
small (sd about 0.12–0.15 near x=1), so the noise shrinks as x grows.

## Validation
Repeated K-fold CV (5-fold × 3–4 repeats; then 10-fold × 3 repeats for the two finalists). Each
model is refit by maximum likelihood (BFGS from several initialisations) on every training fold.
The score is the mean held-out NLL.

## Candidates (CV NLL)
- 2-component Normal mixture, polynomial means/log-sds, sigmoid weight (various degrees): 0.343–0.40
- Component 1 mean = a + b·x + c·sin(w·x + p), log-linear sd; component 2 with a polynomial mean
  (degree / sd-degree / weight-degree):
  - linear mean, constant weight: 0.36–0.40
  - linear mean, log-linear sd, logistic weight: 0.317 (5-fold), 0.312 (10-fold)
  - quadratic mean: 0.323
  - **cubic mean: 0.306 (5-fold), 0.301 (10-fold)**  <- chosen
  - extra sd/weight flexibility (quadratic log-sd, quadratic weight logit, etc.): 0.32–0.35, worse
- A logistic weight with a hard onset near x=0.62–0.85 was worse than the soft logistic weight.

## Final program
Mixture of 2 Normals:
- comp 1: mean = sinusoid + linear trend; sd = exp(linear in x)
- comp 2: mean = cubic in x; sd = exp(linear in x)
- weight of comp 2 = sigmoid(linear in x), rising from 0.22 to 0.53

The fit uses 15 constants. Its training NLL is 0.237 and its CV NLL is about 0.30. It was chosen
because it had the best CV in both 5-fold and 10-fold runs. The CV gap between candidates is
small (about 0.01), so I round the expectation up slightly to allow for selection optimism.

expected_nll: 0.31
