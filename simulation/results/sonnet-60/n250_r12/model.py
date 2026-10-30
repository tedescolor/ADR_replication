from adr.dist import Normal, exp


def model(x):
    t = x[0]
    mu = -0.041798 + t * (-7.95265679 + t * (23.15310482 + t * -14.74753528))
    s = exp(-0.79545325 + t * (-3.33108681 + t * 4.96107349))
    return Normal(mu, s)
