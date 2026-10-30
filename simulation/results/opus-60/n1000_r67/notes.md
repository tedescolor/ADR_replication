# Notes

## Data
1000 rows, x0 roughly uniform on [0,1]. y follows a single noisy curve for x0 < ~0.44
(it dips to about -0.8 at x0≈0.28). After that it splits into two branches of roughly
equal weight: one rises almost linearly to 2.5 at x0=1, the other stays near 0.1 and then
drops to -1. Noise SD shrinks from ~0.35 at x0=0 to ~0.1 at x0=1. The upper branch is
tighter than the trunk.

## Candidates (5-fold CV mean NLL; each fold's fit starts from the full-data fit)
| model | #params | train | CV |
|---|---|---|---|
| 2-normal mixture, free polys (lower deg 6, upper 3, log-sd 1/1), sigmoid gate | 18 | 0.163 | 0.180 |
| same, lower deg 8, log-sd 2/1 | 21 | 0.157 | 0.177 |
| same, lower deg 10 | 23 | 0.156 | 0.178 |
| trunk+branch: m2 = m1 + relu(x-c)*quad, shared sd, constant w | 17 | 0.158 | 0.178 |
| same, Student-t components | 18 | 0.159 | 0.179 |
| **trunk+branch, separate log-sd quadratics, step gate at c** | 21 | 0.152 | **0.171** |
| same, log-sd cubic | 23 | 0.147 | 0.172 |
| same, trunk deg 10 | 23 | 0.148 | 0.189 |
| same, gap cubic / quartic | 22/23 | 0.152/0.147 | 0.177/0.180 |
| same, log-sd linear | 19 | 0.162 | 0.178 |

Differences under ~0.005 are within CV noise.

## Final program
A mixture of two normals:
- The trunk/lower branch m1 is a degree-8 polynomial in z = 2*x0-1.
- The upper branch is m2 = m1 + r*(a + b r + c r^2), where r = max(x0 - 0.439, 0).
- Each component's log-SD is a quadratic in z.
- The upper-branch weight is 0.47 * sigmoid(100*(x0 - 0.439)).

I softened the gate from a hard step to slope 100. That costs 0.002 in train NLL but
makes the model more robust near the split. The program has 25 constants. Train NLL
(as written) is 0.154.

Expected NLL: CV gave 0.171. Warm-starting the folds makes that slightly optimistic,
and I also softened the gate, so I add a small margin.

expected_nll: 0.176
