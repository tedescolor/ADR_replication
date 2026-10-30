from adr.dist import Normal, exp


def model(x):
    t = x[0]
    m = -0.10878 + t * (-6.50119 + t * (18.35590 - 10.70026 * t))
    s = exp(-0.81609 + t * (-3.04702 + 4.60780 * t))
    return Normal(m, s)
