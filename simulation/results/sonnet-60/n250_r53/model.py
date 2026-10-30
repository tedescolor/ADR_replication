from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    z = -3.173 + 4.141 * t
    p = sigmoid(z)
    ma = -0.04644 + t * (-7.36 + t * (22.18 + t * -16.0))
    mb = -2.639 + 5.364 * t
    sa = exp(-0.8447 + -1.262 * t)
    sb = exp(-1.584 + -0.3652 * t)
    return Mixture([1 - p, p], [Normal(ma, sa), Normal(mb, sb)])
