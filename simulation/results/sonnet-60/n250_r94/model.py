from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    t = x[0]
    ma = 0.2095 + t * (-8.9293 + t * (22.3217 - 11.0368 * t))
    mb = -2.4092 + t * (8.3809 - 6.8584 * t)
    sa = exp(-0.8203 - 1.2859 * t)
    sb = exp(-1.3994 - 0.6161 * t)
    w = sigmoid(-3.2725 + 4.0224 * t)
    return Mixture([1 - w, w], [Normal(ma, sa), Normal(mb, sb)])
