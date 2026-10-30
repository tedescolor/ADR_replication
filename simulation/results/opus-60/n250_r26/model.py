from adr.dist import Normal, Mixture, exp, maximum


def model(x):
    z = x[0] - 0.5
    lo = -0.13303733 + z * (2.63378748 + z * (-3.39293296 + z * (-14.20742505 + z * 5.7621244)))
    h = maximum(0.0, x[0] - 0.38)
    up = lo + 0.03908893 * h + 9.33104153 * h * h
    sl = exp(-1.62521122 - 1.2515385 * z)
    su = exp(-1.45463494 - 1.72343567 * z)
    return Mixture([0.448969, 0.551031], [Normal(lo, sl), Normal(up, su)])
