from adr.dist import Normal, Mixture, exp, maximum


def model(x):
    t = x[0]
    up = (((3.6741 * t - 18.2295) * t + 26.2091) * t - 9.3295) * t + 0.1916
    r = maximum(t - 0.5004, 0.0)
    lo = up - (6.6629 * r + 3.6373) * r
    s = exp(-1.3328 * t - 0.8605)
    return Mixture([0.5, 0.5], [Normal(up, s), Normal(lo, s)])
