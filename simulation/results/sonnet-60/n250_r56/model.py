from adr.dist import Normal, exp


def model(x):
    t = x[0] - 0.5
    mu = -0.065358 + t * (4.2256 + t * (0.70154 + t * -16.268))
    s = exp(-1.1683 + t * (1.5507 + t * 4.313))
    return Normal(mu, s)
