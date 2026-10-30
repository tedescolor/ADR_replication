# Notes

## Data exploration
- y has no linear relation to x (all correlations < 0.03). (y0, y1) forms a ring/disk: the angle
  theta = atan2(y1, y0) is close to uniform, and the radius r = |y| depends strongly on x.
- r against linear indices of x is Y-shaped. Local histograms show sharp peaks near r≈1 and r≈3.2
  plus a middle component whose location is linear in x. So r | x is a mixture of experts.
- I tested a noisy-ellipse model, y = M(x)(cos φ, sin φ) + noise with M linear in x, fitted by
  quadrature likelihood. 5-fold CV NLL was 3.476, worse than the isotropic radial mixtures, so I
  dropped it. There is a weak cos/sin(2θ) dependence that I ignored.

## Radial models (5-fold CV on r; 2D NLL = r-NLL + log 2π + mean log r, assuming uniform angle)
| radial model                                        | CV 2D NLL |
|-----------------------------------------------------|-----------|
| single normal, mean and log-sd linear in x          | 3.766     |
| sqrt/tanh of the norm of two projections, het. sd   | 3.548     |
| 2-comp MoE (logistic gate, linear means and log-sds)| 3.396     |
| 3-comp MoE (softmax gates linear + (w·x)^2; comp1 constant, comp2 linear, comp3 linear; constant sds) | **3.295** |

## Representation as an adr distribution
The joint density f(r)/(2πr) cannot be written directly, so each radial normal component
N(R, s) is drawn as a ring of 25 bivariate Gaussians at evenly spaced angles. Each Gaussian has
radial sd s and tangential sd sqrt(s² + (0.3·h·R)²), with h = 2π/25. The rings are nested
Mixtures (5 fans × 5 blobs), built by a shared `spread` helper so there are no loops, and stay
under 500 AST nodes (453). In numpy tests on the 2-comp model the discretisation cost about
0.001–0.003 nats at K=25. For the final model, the in-sample NLL of the actual model.py on all
5000 rows is 3.2903, against 3.2847 for the ideal radial density, so the gap is about 0.006.

## Final program
A 3-component radial mixture of experts, with each component rendered as a ring of 25
Gaussians. It uses 43 constants and 453 AST nodes.
Validation: the 5-fold CV radial NLL converts to a 2D NLL of 3.295. Adding the ≈0.006
representation gap gives ≈3.301. The parameters are fitted on all data (35 radial parameters
for 5000 rows; the train–CV gap was ≈0.01).

expected_nll: 3.31
