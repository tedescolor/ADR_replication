from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    t = x[0]
    ma = ((-15.87 * t + 22.65) * t - 7.9) * t - 0.008474
    mb = ((-11.42 * t + 22.23) * t - 8.492) * t + 0.1338
    sa = exp(-1.001 - 1.056 * t)
    sb = exp(-0.9173 - 0.9835 * t)
    w = sigmoid(-1.848 + 1.936 * t)
    return Mixture([w, 1 - w], [Normal(ma, sa), Normal(mb, sb)])
