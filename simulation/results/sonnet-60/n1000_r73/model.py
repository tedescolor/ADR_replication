from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    u = x[0] - 0.5
    z = -1.1298 + u * (10.324 + u * (-18.347))
    ma = 0.040107 + u * (5.0581 + u * (4.2279 + u * (-10.233 + u * (2.9496))))
    mb = -0.12852 + u * (3.7508 + u * (-10.833))
    sa = exp(-1.4785 + u * (-1.319 + u * (-0.74443)))
    sb = exp(-1.5115 + u * (-1.3068))
    q = sigmoid(z)
    return Mixture([1 - q, q], [Normal(ma, sa), Normal(mb, sb)])
