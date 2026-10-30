# Notes

Data: 1000 rows, x0 in (0,1). The scatter shows two branches: at small x they overlap
(wide, roughly unimodal cloud around -0.5), and for x > ~0.55 they split into an upper
branch rising almost linearly to ~2.5 at x=1 and a lower branch falling to ~-1. Noise
shrinks as x grows.

Model family: a mixture of 2 Normals. Each component has a polynomial mean in z = 2x-1
and a log-linear-polynomial sd. The mixture weights are softmax of polynomials. Fitted by
maximum likelihood (BFGS, several starts).

Validation: 5-fold CV (one fixed random split), mean held-out NLL. Spec is
(mean degree, log-sd degree, weight degree, K):
- 3,2,0,2: 0.2094 | 4,2,0,2: 0.2058 | 5,2,0,2: 0.2075 | 6,2,0,2: 0.2082
- 4,0,0,2: 0.2742 | 3,1,0,2: 0.2085 | **4,1,0,2: 0.2026** | 5,1,0,2: 0.2069
- 4,3,0,2: 0.2104 | 4,4,0,2: 0.2118 | 4,3,1,2: 0.2099 | 5,3,0,2: 0.2121
- 4,2,0,3 (3 components): 0.2268

Final: 4,1,0,2, refit on all data (train NLL 0.187, 15 fitted parameters, about 19 literals).
The weight is about 0.5 for each branch. Differences between nearby specs are about 0.005,
which is within CV noise, so the CV minimum is a little optimistic. I used the CV value and
added a small allowance for that selection bias.

expected_nll: 0.21
