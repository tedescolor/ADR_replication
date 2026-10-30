from adr.dist import Normal, exp


def model(x):
    t = x[0]
    m = -0.217 + t * (-6.218 + t * (19.2 + t * -12.2))
    s = exp(-0.7123 + t * (-3.375 + t * 4.877))
    return Normal(m, s)
