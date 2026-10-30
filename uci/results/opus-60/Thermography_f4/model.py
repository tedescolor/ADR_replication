from adr.dist import Normal, SkewNormal, Mixture, exp, log, sqrt, tanh, sigmoid, maximum, minimum


def model(x):
    """y = x24 - D, D >= 0: mixture of a sharp near-x24 component and a wide
    half-normal-like component; the mixing weight depends on how far x24
    sits above the other members x1..x23. A tiny broad normal guards the tails."""
    o = x[1:24]
    g = x[24] - max(o)
    m24 = sum(x[1:]) / 24.0
    sd = sqrt(maximum(sum((x[1:] - m24) ** 2) / 24.0, 1e-6))
    dm = x[24] - sum(o) / 23.0
    loc = x[24] - 0.009
    w = sigmoid(20.7273 - 2.8258 * tanh(10.0 * g) + 6.8912 * tanh(40.0 * g) - 25.8699 * tanh(100.0 * g))
    near = SkewNormal(loc, exp(-3.2193), -exp(1.2565))
    far = SkewNormal(loc, exp(minimum(-0.5071 + 0.1557 * log(sd) + 0.5439 * dm, 3.0)), -exp(8.0))
    wide = Normal(x[24] - 0.5, 1.0)
    return Mixture([0.998 * (1.0 - w), 0.998 * w, 0.002], [near, far, wide])
