from adr.dist import Normal, exp


def model(x):
    t = x[0]
    m = -0.02411746 + t * (-7.902784 + t * (22.98453533 + t * -14.69628335))
    s = exp(-0.81432947 + t * (-3.20692776 + t * 4.81765181))
    return Normal(m, s)
