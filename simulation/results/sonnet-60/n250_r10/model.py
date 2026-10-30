from adr.dist import Normal, exp


def model(x):
    t = x[0]
    mu = 0.0142 + t * (-8.1204 + t * (24.1365 - 15.9256 * t))
    sd = exp(-0.7195 + t * (-4.2014 + 5.9201 * t))
    return Normal(mu, sd)
