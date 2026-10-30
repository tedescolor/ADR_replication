from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    lo = 0.2468 + t * (-10.52 + t * (34.73 + t * (-34.6 + t * 9.089)))
    hi = -2.312 + 4.945 * t
    w = sigmoid(-4.142 + 4.83 * t)
    return Mixture([1 - w, w], [Normal(lo, exp(-0.8921 - 1.236 * t)),
                                Normal(hi, exp(-1.83 - 0.2506 * t))])
