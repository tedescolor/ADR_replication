from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    t = x[0]
    ma = ((((12.63 * t + -40.35) * t + 36.94) * t + -10.31) * t + 0.1075)
    mb = 4.906 * t + -2.294
    sa = exp(-0.8449 + -1.281 * t)
    sb = exp(-0.2495 + -2.126 * t)
    w = sigmoid(-3.253 + 4.048 * t)
    return Mixture([1 - w, w], [Normal(ma, sa), Normal(mb, sb)])
