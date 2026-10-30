from adr.dist import Mixture, SkewNormal, sigmoid, maximum, sqrt, exp, log


def model(x):
    t = x[24]
    o = x[1:24]
    m = maximum(t - o.max(), 0.0)
    sd = maximum(o.std(), 0.01)
    w = sigmoid(-2.8572 + 8.256 * sqrt(m))
    near = SkewNormal(t - 0.0082, 0.0399, -3.015)
    below = SkewNormal(t - 0.0849, exp(-0.0678 + 0.4048 * log(sd)), -22.0534)
    return Mixture([w, 1 - w], [near, below])
