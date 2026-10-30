from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    z = x[0] - 0.5
    ma = -0.02 + z * (5.122 + z * (4.771 - 9.752 * z))
    mb = -0.364 + z * (5.33 - 13.153 * z)
    s = exp(-1.474 + z * (-1.393 - 0.985 * z))
    w = 0.5 * sigmoid(50 * (x[0] - 0.539))
    return Mixture([1 - w, w], [Normal(ma, s), Normal(mb, s)])
