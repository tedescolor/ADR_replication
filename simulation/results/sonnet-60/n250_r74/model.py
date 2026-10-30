from adr.dist import Normal, exp


def model(x):
    t = x[0]
    m = -0.1498 + t * (-7.061 + t * (21.96 + t * -14.53))
    s = exp(-0.8569 + t * (-3.163 + t * 4.801))
    return Normal(m, s)
