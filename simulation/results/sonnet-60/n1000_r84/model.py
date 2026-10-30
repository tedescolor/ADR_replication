from adr.dist import Mixture, Normal, exp, maximum, sigmoid


def model(x):
    t = x[0]
    m1 = ((((9.8381 * t + -34.2898) * t + 32.9865) * t + -9.5766) * t + 0.0696)
    h = maximum(t - 0.3726, 0.0)
    m2 = m1 + (0.2055 + 8.6378 * h) * h
    w = sigmoid(-0.6953 + 0.7972 * t)
    return Mixture([w, 1.0 - w], [Normal(m1, exp(-0.8731 + -1.2571 * t)),
                                  Normal(m2, exp(-0.9279 + -1.2647 * t))])
