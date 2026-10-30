from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    ma = 0.3244 + t * (-11.9468 + t * (41.0468 + t * (-44.3884 + t * 13.9519)))
    mb = -2.3946 + 5.0316 * t
    pb = sigmoid(-2.826 + 3.395 * t)
    sa = exp(-0.7728 + -1.5642 * t)
    sb = exp(-1.3961 + -0.6941 * t)
    return Mixture([1 - pb, pb], [Normal(ma, sa), Normal(mb, sb)])
