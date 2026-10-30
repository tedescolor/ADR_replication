from adr.dist import Normal, Mixture, exp, maximum


def model(x):
    t = 2.0 * x[0] - 1.0
    g = 0.02222 + t * (2.53328 + t * (1.52165 - 1.13176 * t))
    r = maximum(t + 0.20751, 0.0)
    s = exp(-1.55304 - 0.6516 * t)
    return Mixture([0.47912, 0.52088],
                   [Normal(g - 0.286 * r * r, s), Normal(g - 2.74381 * r * r, s)])
