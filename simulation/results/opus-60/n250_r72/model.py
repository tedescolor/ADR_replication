from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    """Bifurcation: one cubic curve for small x; past x ~ 0.34 a linear upper branch takes
    half the mass (steep sigmoid switch). Both branches share a log-linear scale."""
    t = x[0]
    m1 = ((-14.451 * t + 20.178) * t - 6.707) * t - 0.09
    m2 = 5.078 * t - 2.45
    s = exp(-1.413 * t - 0.815)
    w = 0.5 * sigmoid(50.0 * (t - 0.34))
    return Mixture([1 - w, w], [Normal(m1, s), Normal(m2, s)])
