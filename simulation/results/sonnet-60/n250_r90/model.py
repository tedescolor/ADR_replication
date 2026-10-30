from adr.dist import Normal, exp


def model(x):
    t = x[0]
    mu = 0.15744224 + t * (-9.04366998 + t * (25.67662952 + t * -16.65111767))
    s = exp(-0.60519916 + t * (-4.26800672 + t * 5.76954042))
    return Normal(mu, s)
