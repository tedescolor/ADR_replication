from adr.dist import Normal, exp


def model(x):
    t = x[0]
    mu = (((-15.7296 * t + 24.0644) * t + -8.348) * t) + 0.1021
    ls = (5.4057 * t + -4.078) * t + -0.5546
    return Normal(mu, exp(ls))
