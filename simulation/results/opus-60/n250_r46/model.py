from adr.dist import Normal, Mixture, exp, maximum, sigmoid


def model(x):
    t = x[0]
    h = maximum(t - 0.442, 0.0)
    tr = -0.0816 - 5.7037 * t + 11.6704 * t ** 2
    m1 = tr + 1.5338 * h - 13.437 * h ** 2
    m2 = tr - 0.2653 * h - 21.5248 * h ** 2
    s = exp(-0.9887 - 1.1685 * t)
    w = sigmoid(0.0505)
    return Mixture([w, 1.0 - w], [Normal(m1, s), Normal(m2, s)])
