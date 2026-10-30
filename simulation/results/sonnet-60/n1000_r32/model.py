from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    lo = (((12.601 * t - 39.3947) * t + 35.4402) * t - 9.6546) * t + 0.0498
    up = (-1.6121 * t + 7.4349) * t - 3.2659
    s1 = exp(-0.9135 - 1.2305 * t)
    s2 = exp(-0.9088 - 1.2757 * t)
    w = sigmoid(-3.3515 + 3.7086 * t)
    return Mixture([1 - w, w], [Normal(lo, s1), Normal(up, s2)])
