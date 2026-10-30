from adr.dist import Normal, Mixture, exp, sin, maximum


def model(x):
    """Smooth sine-trend curve that splits at t ~ 0.497 into two equally likely
    branches (quadratic offsets in the distance past the split); shared
    Gaussian noise with log-sd cubic in t."""
    t = x[0]
    f = 0.81156 - 0.71474 * t + 1.45290 * sin(4.47060 * t + 3.76436)
    r = maximum(t - 0.49733, 0.0)
    s = exp(((-2.81358 * t + 3.25016) * t - 2.15100) * t - 0.78436)
    up = Normal(f + (-1.76196 + 7.72202 * r) * r, s)
    down = Normal(f + (-5.26027 + 0.89323 * r) * r, s)
    return Mixture([0.5, 0.5], [up, down])
