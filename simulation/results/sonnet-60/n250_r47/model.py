from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    u = x[0] - 0.5
    ma = -0.024 + u * (5.04 + u * (5.169 - 10.083 * u))
    mb = 0.124 + u * (1.458 - 7.193 * u)
    sa = exp(-1.567 - 1.392 * u)
    sb = exp(-1.421 - 1.216 * u)
    w = sigmoid(1.144 - 3.041 * u)
    return Mixture([w, 1 - w], [Normal(ma, sa), Normal(mb, sb)])
