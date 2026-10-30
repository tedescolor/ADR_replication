from adr.dist import Normal, Mixture, exp, maximum, sigmoid


def model(x):
    t = 2 * x[0] - 1
    f = -0.018 + t * (2.514 + t * (1.268 - 1.281 * t))
    d = maximum(x[0] - 0.505, 0.0) * (3.696 + 3.327 * t)
    s = exp(-1.558 - 0.647 * t)
    w = sigmoid(-0.012)
    return Mixture([w, 1 - w], [Normal(f, s), Normal(f - d, s * exp(0.02))])
