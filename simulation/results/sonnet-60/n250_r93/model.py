from adr.dist import Normal, exp


def model(x):
    t = x[0]
    mu = ((-9.026 * t + 15.565) * t + -5.4134) * t + -0.1691
    sigma = exp((6.1321 * t + -4.6) * t + -0.5752)
    return Normal(mu, sigma)
