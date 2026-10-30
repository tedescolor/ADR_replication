"""Mixture of a smooth curve (polynomial mean, log-polynomial scale) and a line-like
branch that switches on at x0 ~ 0.315 with constant weight."""
from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = 2.0 * x[0] - 1.0
    m0 = ((((0.58678) * t + -1.7997) * t + -0.94005) * t + 1.2767) * t + -0.11183
    m1 = ((-0.43007) * t + 2.9328) * t + 0.051776
    s0 = exp(((-0.29574) * t + -0.67843) * t + -1.4005)
    s1 = exp((-0.65842) * t + -1.5112)
    w = 0.45958 * sigmoid(500.0 * (x[0] - 0.31509))
    return Mixture([1.0 - w, w], [Normal(m0, s0), Normal(m1, s1)])
