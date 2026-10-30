from adr.dist import Normal, Mixture, exp, maximum, sigmoid


def model(x):
    t = x[0]
    w = sigmoid(0.164552)
    up = Normal((-0.131315 + t * (-5.7669 + t * (16.2884 + t * -7.84643))), exp((-0.941267 + t * -1.28248)))
    lo = Normal((0.207423 + t * (-12.4212 + t * (41.55 + t * (-43.2678 + t * 12.9539)))), exp((-0.941267 + t * -1.28248)))
    return Mixture([w, 1 - w], [up, lo])
