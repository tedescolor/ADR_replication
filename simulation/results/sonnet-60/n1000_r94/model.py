from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    u = x[0] - 0.5
    ma = (((12.77 * u - 15.35) * u - 4.51) * u + 2.773) * u - 0.1164
    mb = ((-9.97 * u + 4.836) * u + 4.955) * u + 0.03554
    sa = exp((-1.863 * u - 1.247) * u - 1.353)
    sb = exp(-1.694 * u - 1.54)
    z = sigmoid(0.1764 - 0.2079 * u)
    return Mixture([1 - z, z], [Normal(ma, sa), Normal(mb, sb)])
