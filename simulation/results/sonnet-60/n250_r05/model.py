from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    ma = -0.15419378 + t * (-6.51598696 + t * (20.30232177 + t * -14.74238376))
    mb = -2.12350843 + 4.71866828 * t
    sa = exp(-0.94531081 - 1.1141533 * t)
    w = sigmoid(-2.87479704 + 3.22542473 * t)
    return Mixture([1 - w, w], [Normal(ma, sa), Normal(mb, 0.139824)])
