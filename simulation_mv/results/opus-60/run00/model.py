"""Bar (two blobs at +-mb along a covariate-dependent axis) + isotropic noisy ring.

The ring is built as p(y0) p(y1 | y0): y0 is a 5-normal approximation of the noisy
arcsine marginal, y1 | y0 is +-sqrt(rr^2 - y0^2) with slope-adjusted noise."""
from adr.dist import (MultivariateNormal, Normal, Sequential, Mixture, sigmoid, sqrt,
                      maximum, cos, sin, tanh, dot)


def model(x):
    al = 0.0005 + dot(x, [-0.7226, 0.4774, -0.3601, 0.2397, -0.54])
    w = sigmoid(0.0019 + dot(x, [-1.1844, 0.7914, -0.563, 0.3961, -0.8846]))
    mb = 1.9849 + 1.211 * tanh(0.0067 + dot(x, [-0.3977, 0.8614, -0.2395, 0.556, 0.3063]))
    rr = 1.9416 + 1.505 * tanh(-0.0185 + dot(x, [0.853, -0.5791, 0.4283, -0.2955, 0.6388]))
    ca, sa = cos(al), sin(al)
    vv = 0.0534
    c, e = -0.0042 * ca, -0.0042 * sa
    cov = [[vv + c * ca, c * sa], [c * sa, vv + e * sa]]
    bar = Mixture([1, 1], [MultivariateNormal([mb * ca, mb * sa], cov),
                           MultivariateNormal([-mb * ca, -mb * sa], cov)])
    s0 = 0.1876 * rr + 0.518
    s1 = 0.1314 * rr + 0.2014
    s2 = 0.0008 * rr + 0.275
    m1, m2 = 0.6097 * rr, 0.9561 * rr
    w1, w2 = 0.6043, 0.6384
    first = Mixture([1, w1, w1, w2, w2], [Normal(0, s0), Normal(m1, s1), Normal(-m1, s1),
                                          Normal(m2, s2), Normal(-m2, s2)])

    def cond(y0):
        h = sqrt(maximum(rr * rr - y0 * y0, 0))
        s = 0.2742 * rr / sqrt(h * h + 0.2948 * rr)
        return Mixture([1, 1], [Normal(h, s), Normal(-h, s)])

    return Mixture([w, 1 - w], [bar, Sequential(first, cond)])
