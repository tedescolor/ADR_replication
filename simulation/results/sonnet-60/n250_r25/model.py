from adr.dist import Normal, exp


def model(x):
    t = x[0]
    m = -0.15428587 + t * (-6.7752464 + t * (20.91170919 - 13.8255526 * t))
    s = exp(-0.72403797 + t * (-4.08620155 + 5.80325436 * t))
    return Normal(m, s)
