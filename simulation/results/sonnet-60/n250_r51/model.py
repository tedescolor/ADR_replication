from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    ma = -0.294 + t * (-5.646 + t * (17.97 - 13.054 * t))
    mb = 0.0039 + t * (-6.537 + t * (17.359 - 8.247 * t))
    sa = exp(-1.101 - 1.261 * t)
    sb = exp(-0.782 - 1.536 * t)
    w = sigmoid(-0.256)
    return Mixture([w, 1 - w], [Normal(ma, sa), Normal(mb, sb)])
