from adr.dist import Normal, Mixture, abs, maximum, sigmoid


def model(x):
    """Trunk cubic that splits at x=0.4687 into two quadratic branches; shared sd."""
    t = x[0]
    f = ((-11.7193 * t + 22.1175) * t - 8.1207) * t + 0.0334
    h = maximum(t - 0.4687, 0.0)
    lo = f - 3.3501 * h - 5.484 * h * h
    up = f - 0.5764 * h + 1.5768 * h * h
    s = abs(0.2775 * (1.0 - t)) + 0.0853
    w = sigmoid(-0.3685)
    return Mixture([w, 1.0 - w], [Normal(lo, s), Normal(up, s)])
