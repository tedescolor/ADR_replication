from adr.dist import Normal, Mixture, sigmoid, exp


def model(x):
    t = x[0]
    ma = ((-9.984 * t + 20.216) * t - 7.787) * t + 0.063
    mb = (-7.87 * t + 9.904) * t - 2.929
    sa = exp(-0.868 - 1.279 * t)
    sb = exp(-0.32 - 1.905 * t)
    w = sigmoid(-3.038 + 3.9 * t)
    return Mixture([1 - w, w], [Normal(ma, sa), Normal(mb, sb)])
