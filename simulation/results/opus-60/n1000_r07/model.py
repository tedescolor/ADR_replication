"""Two equal-weight Gaussian branches around a common center.

t = 2*x0 - 1; center c(t) = a0 + a1 t + a3 t^3 + a4 t^4; half-gap g(t) = t (b1 + b2 t);
branches c +/- g share the scale s(t) = |s0 + s1 t|.
"""
from adr.dist import Normal, Mixture, abs


def model(x):
    t = 2.0 * x[0] - 1.0
    c = -0.0485 + t * (1.9848 + t * t * (-1.6527 + 0.4786 * t))
    g = t * (0.8172 + 0.9545 * t)
    s = abs(0.2367 - 0.1442 * t) + 0.000001
    return Mixture([0.5, 0.5], [Normal(c + g, s), Normal(c - g, s)])
