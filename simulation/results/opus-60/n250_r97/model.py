from adr.dist import Normal, Mixture, exp, maximum


def model(x):
    v = x[0]
    c = 0.00521 + v * (-4.85101 + v * (-0.6651 + v * (47.9111 + v * (-70.07684 + 28.50368 * v))))
    u = maximum(v - 0.37049, 0.0)
    h = 0.5 * u * (-0.00021 + 8.8645 * u)
    s = exp(-0.90881 - 1.43465 * v)
    return Mixture([0.39102, 0.60898], [Normal(c + h, s), Normal(c - h, s)])
