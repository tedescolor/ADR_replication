from adr.dist import Normal, exp


def model(x):
    t = x[0] - 0.5
    mu = -0.066 + t * (3.8585 + t * (2.0001 + t * -10.9902))
    ls = -1.216 + t * (1.4703 + t * 5.1221)
    return Normal(mu, exp(ls))
