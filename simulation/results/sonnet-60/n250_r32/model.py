from adr.dist import Normal, exp


def model(x):
    t = x[0]
    mu = 0.21769 + t * (-12.551 + t * (45.629 + t * (-54.423 + t * 22.755)))
    s = exp(-0.47896 + t * (-4.5174 + t * 5.8855))
    return Normal(mu, s)
