from adr.dist import Normal, Mixture, sigmoid, exp


def model(x):
    """Two-branch mixture: curved lower branch (quartic mean, log-linear scale) and a
    linear upper branch (constant scale) whose weight grows logistically with x0."""
    t = x[0]
    lo = 0.23157 + t * (-11.991 + t * (43.48 + t * (-49.89 + t * 17.279)))
    s_lo = exp(-0.80382 - 1.0544 * t)
    hi = -2.6662 + 5.3604 * t
    s_hi = exp(-2.2013)
    w = sigmoid(-3.9643 + 5.0068 * t)
    return Mixture([1.0 - w, w], [Normal(lo, s_lo), Normal(hi, s_hi)])
