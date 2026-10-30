from adr.dist import Normal, exp


def model(x):
    t = x[0]
    m = -0.01 + t * (-7.6494 + t * (21.9265 + t * -13.6079))
    s = exp(-1.0761 + t * (-2.4391 + t * 4.2906))
    return Normal(m, s)
