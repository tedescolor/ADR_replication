# Notes

## Data
For x0 < ~0.55, y is a single noisy curve (dips to about -0.8 near x=0.25). After that it
splits into two branches: the upper one rises to about 2.5 at x=1 and the lower one falls to
about -1. The noise sd shrinks with x (about 0.4 on the left, about 0.1 on the right). Within
each branch the residuals look Gaussian: in-sample PIT z has skew about 0 and excess
kurtosis about -0.2.

## Candidates (all are 2-component Gaussian mixtures fitted by MLE with L-BFGS and random restarts)
Validation is 5-fold CV of the mean held-out NLL. I repeated it on 3 random fold
splits (seeds 0, 1, 2) because a single split moves by about 0.01.

1. Each component has its own free polynomial mean in t=2x-1, a polynomial log-sd and a
   polynomial logit weight. Results with (mean coefs, logsd coefs, weight coefs), seed 0:
   (4,3,2) 0.2244, (5,3,2) 0.2195, (6,3,2) 0.2226, (7,3,2) 0.2332, (8,3,2) 0.2346,
   (10,3,2) 0.2372, (6,2,2) 0.2264, (6,3,1) 0.2246, (6,4,2) 0.2227, (5,3,1) 0.2209,
   (5,4,2) 0.2236, (5,2,1) 0.2236.
   Best (5,3,2) over seeds 0/1/2: 0.2195 / 0.2343 / 0.2202, mean 0.2247.
   Higher degrees overfit: on the left, where the two components overlap, they can bend
   freely.
2. **Chosen:** a shared trunk f(t) (quartic) plus branch offsets that switch on through a
   sigmoid gate, h(t)*g_k(t) with quadratic g_k. Each component has its own quadratic log-sd,
   and the logit weight is linear. 21 constants.
   CV over seeds 0/1/2: 0.2183 / 0.2258 / 0.2175, mean 0.2205. Train NLL 0.1946.
   It beat the free-polynomial mixture on all 3 splits, and it varies less
   between splits. Its structure (one curve that bifurcates) matches the data.

## Estimate
The CV mean is 0.2205. The final model is fitted on all 1000 rows (CV trained on 800), which
helps a little, so I estimate about 0.22.

expected_nll: 0.22
