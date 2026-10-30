from adr.dist import Normal, Mixture, exp, maximum


def model(x):
    t = x[0]
    m1 = -0.1336 + t * (-6.5009 + t * (17.7653 + t * -8.5837))
    s = exp(-0.9757 - 1.1573 * t)
    r = maximum(t - 0.5113, 0.0)
    m2 = m1 - r * (2.9116 + 8.9657 * r)
    w = 0.45867
    return Mixture([1 - w, w], [Normal(m1, s), Normal(m2, s)])
