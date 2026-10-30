from adr.dist import Normal, exp


def model(x):
    u = x[0]
    m = 0.088603 + u * (-9.1053 + u * (30.906 + u * (-32.751 + u * 12.303)))
    s = exp(-0.75697 + u * (-3.3754 + u * 4.9201))
    return Normal(m, s)
