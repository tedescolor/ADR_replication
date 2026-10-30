from adr.dist import Normal, exp


def model(x):
    t = x[0] - 0.5
    mu = -0.08779752 + t * (3.90796764 + t * (1.5398367 - 13.97134896 * t))
    s = exp(-1.20026202 + t * (1.59060359 + 4.72322543 * t))
    return Normal(mu, s)
