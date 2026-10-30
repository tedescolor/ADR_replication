from adr.dist import Normal, exp


def model(x):
    t = x[0]
    mu = -0.0112694836 + t * (-7.55724168 + t * (22.2580563 + t * -14.4475319))
    s = exp(-0.696300187 + t * (-3.10933969 + t * 4.49073121))
    return Normal(mu, s)
