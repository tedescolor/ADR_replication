from adr.dist import Normal, exp


def model(x):
    t = x[0]
    m = 0.13240345 + t * (-10.56655587 + t * (36.06009953 + t * (-37.91715057 + t * 13.41773131)))
    s = exp(-0.70208978 + t * (-3.69485938 + t * 5.2143729))
    return Normal(m, s)
