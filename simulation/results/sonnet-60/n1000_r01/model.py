from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    ma = 0.117 + t * (-10.62 + t * (38.2 + t * (-42.26 + t * 13.61)))
    mb = -2.297 + 4.91 * t
    sa = exp(-0.9101 + -1.061 * t)
    sb = exp(-0.6899 + -1.522 * t)
    p = sigmoid(-3.781 + 4.446 * t)
    return Mixture([1 - p, p], [Normal(ma, sa), Normal(mb, sb)])
