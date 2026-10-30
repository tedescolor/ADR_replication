from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = 2.0 * x[0] - 1.0
    # main branch: degree-6 mean, log-quadratic scale (in t = 2x - 1)
    m1 = -0.0155455 + t * (2.65813 + t * (1.25778 + t * (-1.7375 + t * (0.209335 + t * (0.347227 + t * -0.236271)))))
    s1 = exp(-1.45443 + t * (-0.739239 + t * -0.284965))
    # second branch, present only for x > 0.417: cubic mean, log-linear scale
    m2 = -0.00133486 + t * (0.93243 + t * (-0.746602 + t * -1.18266))
    s2 = exp(-1.49877 - 0.713356 * t)
    w2 = 0.485263 * sigmoid(1000.0 * (x[0] - 0.417207))
    return Mixture([1.0 - w2, w2], [Normal(m1, s1), Normal(m2, s2)])
