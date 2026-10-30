from adr.dist import Normal, exp


def model(x):
    t = x[0]
    mu = -0.0097808 + t * (-9.0119 + t * (30.614 + t * (-29.585 + t * 8.6394)))
    sd = exp(-0.71812 + t * (-3.6343 + t * 5.1738))
    return Normal(mu, sd)
