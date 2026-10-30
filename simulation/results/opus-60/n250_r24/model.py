from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    """Two-branch Gaussian mixture: upper branch (cubic) and lower branch (quartic)
    coincide for x < 0.5 and split above; shared log-linear noise scale."""
    t = x[0]
    m1 = -0.06056 + t * (-6.55243 + t * (18.03613 + t * -8.93176))
    m2 = -0.01597 + t * (-10.92079 + t * (39.68921 + t * (-43.53293 + t * 13.85469)))
    s = exp(-0.92458 - 1.27255 * t)
    w = sigmoid(0.29122)
    return Mixture([w, 1.0 - w], [Normal(m1, s), Normal(m2, s)])
