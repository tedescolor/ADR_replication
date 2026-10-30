from adr.dist import Normal, Mixture, exp, abs, sigmoid


def model(x):
    t = x[0]
    # main branch: quartic mean, noise sd shrinking linearly in x
    m1 = -0.0432 + t * (-7.6743 + t * (22.6478 + t * (-15.4176 + t * 3.0475)))
    s1 = abs(0.3755 - 0.3126 * t) + 0.001
    # lower branch appearing for x > ~0.47
    m2 = -4.8054 + t * (14.432 - 10.5491 * t)
    s2 = exp(-0.3321 - 2.1072 * t)
    w = sigmoid(-0.1999) * sigmoid(60 * (t - 0.4722))
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
