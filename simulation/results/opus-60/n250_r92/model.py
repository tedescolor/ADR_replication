from adr.dist import Normal, Mixture, exp


def model(x):
    t = x[0]
    m1 = 0.1146 + t * (-9.518 + t * (24.6912 - 12.8789 * t))
    m2 = -0.1358 + t * (-4.4783 + t * (14.7684 - 11.2059 * t))
    s = exp(-1.0196 - 1.2008 * t)
    return Mixture([0.5, 0.5], [Normal(m1, s), Normal(m2, s)])
