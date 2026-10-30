from adr.dist import Normal, exp


def model(x):
    t = x[0]
    m = ((-13.212 * t + 21.677) * t + -7.7073) * t + 0.000872
    s = exp((5.7284 * t + -3.978) * t + -0.78691)
    return Normal(m, s)
