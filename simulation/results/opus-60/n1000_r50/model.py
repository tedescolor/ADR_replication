from adr.dist import Normal, Mixture, sigmoid, exp, maximum


def model(x):
    t = 2.0 * x[0] - 1.0
    r = maximum(x[0] - 0.35204, 0.0)
    base = ((-0.77124 * t + 1.45384) * t + 2.08418) * t - 0.15138
    mu1 = base + 1.98423 * r - 3.35394 * r * r
    mu2 = base + 2.03421 * r - 11.77442 * r * r
    s1 = exp((-0.31337 * t - 0.75872) * t - 1.36012)
    s2 = exp((0.02963 * t - 0.43842) * t - 1.64397)
    w = sigmoid(-0.85087 * t + 0.42587)
    return Mixture([w, 1.0 - w], [Normal(mu1, s1), Normal(mu2, s2)])
