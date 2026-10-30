from adr.dist import Normal, exp


def model(x):
    t = x[0]
    m = -0.09898 + t * (-7.34109 + t * (21.86471 - 13.98079 * t))
    s = exp(-0.77701 + t * (-2.95812 + 4.53074 * t))
    return Normal(m, s)
