"""Two-branch normal mixture: y = g(x) +/- h(x) + noise, equal weights.
g: quartic; h: hinge-quadratic splitting at x = c; log sd quadratic, shared."""
from adr.dist import Normal, Mixture, exp, maximum


def model(x):
    x = x[0]
    g = ((((4.57374) * x + -21.6719) * x + 26.321) * x + -8.50389) * x + 0.0552952
    u = maximum(x - 0.519432, 0.0)
    h = u * (2.18519 + 2.97185 * u)
    s = exp(((-1.02357) * x + -0.404678) * x + -0.932627)
    return Mixture([0.5, 0.5], [Normal(g + h, s), Normal(g - h, s)])
