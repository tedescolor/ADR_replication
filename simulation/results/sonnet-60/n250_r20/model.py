from adr.dist import Normal, exp


def model(x):
    t = x[0]
    m = -0.00273 + t * (-7.60936 + t * (21.45043 - 13.21847 * t))
    s = exp(-0.89878 + t * (-2.74818 + 4.40367 * t))
    return Normal(m, s)
