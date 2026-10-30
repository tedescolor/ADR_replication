from adr.dist import Normal, exp


def model(x):
    u = x[0]
    mu = ((-10.073 * u + 17.279) * u + -6.1333) * u + -0.096202
    ls = (4.5302 * u + -3.1353) * u + -0.68502
    return Normal(mu, exp(ls))
