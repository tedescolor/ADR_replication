from adr.dist import Normal, exp


def model(x):
    t = x[0]
    m = -0.072 + t * (-7.1523 + t * (20.8013 + t * -12.9765))
    s = exp(-0.5485 + t * (-4.5383 + t * 6.0292))
    return Normal(m, s)
