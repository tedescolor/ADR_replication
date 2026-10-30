from adr.dist import Mixture, Normal, StudentT, exp, sigmoid


def model(x):
    """Two-branch mixture: a main Student-t curve (quartic mean, log-linear scale) and,
    for x0 > ~0.55, an upper linear Normal branch taking half the mass."""
    t = x[0]
    m1 = 0.374 + t * (-13.364 + t * (47.86 + t * (-55.591 + t * 19.794)))
    s1 = exp(-1.004 - 1.302 * t)
    m2 = -2.31 + 4.897 * t
    s2 = exp(0.189 - 2.719 * t)
    w = 0.5 * sigmoid(100.0 * (t - 0.546))
    return Mixture([1.0 - w, w], [StudentT(8.7, m1, s1), Normal(m2, s2)])
