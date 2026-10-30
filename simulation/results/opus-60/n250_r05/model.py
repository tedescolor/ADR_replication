from adr.dist import Normal, Mixture, sin, exp, sigmoid


def model(x):
    """Two-branch mixture: a sinusoidal main curve plus a linear upper branch
    that switches on for x0 > ~0.5 and takes ~47% of the mass."""
    t = x[0]
    m1 = 0.46094 - 1.65367 * t + 0.94987 * sin(5.35805 * t + 3.84515)
    s1 = exp(-1.16329 + 0.46298 * t - 1.78078 * t * t)
    m2 = -2.13528 + 4.73531 * t
    s2 = exp(-1.96788)
    w = sigmoid(9.60000 * (t - 0.50611)) * sigmoid(-0.13519)
    return Mixture([1.0 - w, w], [Normal(m1, s1), Normal(m2, s2)])
