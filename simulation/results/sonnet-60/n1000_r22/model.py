from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    a = 13.826 * t**4 - 42.485 * t**3 + 38.064 * t**2 - 10.424 * t + 0.079
    sa = exp(-0.877 - 1.072 * t - 0.29 * t * t)
    b = -3.599 + 8.372 * t - 2.247 * t * t
    sb = exp(-1.011 - 1.162 * t)
    w = sigmoid(5.042 * (t - 0.831))
    return Mixture([1 - w, w], [Normal(a, sa), Normal(b, sb)])
