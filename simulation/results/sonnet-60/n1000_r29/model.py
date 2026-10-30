from adr.dist import Normal, exp


def model(x):
    u = x[0] - 0.5
    m = -0.05189334 + u * (4.10630162 + u * (1.02397774 - 13.8729391 * u))
    s = exp(-1.19920212 + u * (1.54548389 + 4.85717989 * u))
    return Normal(m, s)
