from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = ((-9.397 * t + 19.088) * t - 7.111) * t - 0.036
    s1 = exp(-0.954 - 1.333 * t)
    m2 = (-9.927 * t + 13.132) * t - 4.16
    s2 = exp(0.014 - 2.281 * t)
    w = sigmoid(-3.637 + 4.516 * t)
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
