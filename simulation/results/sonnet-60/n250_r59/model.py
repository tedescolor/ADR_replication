from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = (((11.95618235 * t - 40.91254892) * t + 39.42524004) * t - 11.80803976) * t + 0.30995107
    m2 = -2.07802757 + 4.690277 * t
    s1 = exp(-0.84213147 - 1.47041489 * t)
    s2 = exp(-0.99441795 - 1.04523013 * t)
    w = sigmoid(-4.03033696 + 4.65656905 * t)
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
