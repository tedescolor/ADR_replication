from adr.dist import Normal, exp


def model(x):
    t = x[0]
    mu = ((-12.705 * t + 20.926) * t - 7.4388) * t + 0.015668
    ls = (4.3909 * t - 2.9131) * t - 0.73805
    return Normal(mu, exp(ls))
