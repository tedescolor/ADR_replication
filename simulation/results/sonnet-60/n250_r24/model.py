from adr.dist import Normal, exp


def model(x):
    t = x[0]
    m = -0.067 + t * (-7.6735 + t * (22.5404 + t * -14.0777))
    s = exp(-0.785 + t * (-3.5137 + t * 5.0768))
    return Normal(m, s)
