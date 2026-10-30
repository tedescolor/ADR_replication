from adr.dist import Normal, exp


def model(x):
    t = x[0]
    m = ((-15.79 * t + 25.29) * t - 9.095) * t + 0.124
    s = exp((4.717 * t - 3.108) * t - 0.8448)
    return Normal(m, s)
