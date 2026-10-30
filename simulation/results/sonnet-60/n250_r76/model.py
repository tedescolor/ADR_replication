from adr.dist import Normal, exp


def model(x):
    t = x[0]
    m = -0.00125698 + t * (-7.87489981 + t * (22.6948417 + t * -14.3568274))
    s = exp(-0.799176685 + t * (-3.39825524 + t * 5.05678595))
    return Normal(m, s)
