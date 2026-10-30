from adr.dist import Normal, exp


def model(x):
    t = x[0] - 0.5
    m = 0.02837 + t * (3.738 + t * (-2.809 + t * (-11.57 + t * 22.57)))
    s = exp(-1.361 + t * (1.63 + t * 5.803))
    return Normal(m, s)
