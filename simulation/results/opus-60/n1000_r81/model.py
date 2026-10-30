from adr.dist import Normal, Mixture, sigmoid, maximum


def model(x):
    """Two-branch mixture: a wavy lower branch everywhere, plus an upper branch for x > 0.526."""
    t = x[0] - 0.5
    ml = -0.0747977 + t * (2.9554463 + t * (-7.3101826 + t * (-19.7230143 + t * (48.2597548 + t * (17.6202417 + t * -107.3526900)))))
    mu = 0.0995771 + t * (3.8076351 + t * (8.8276663 + t * -13.9248568))
    sl = maximum(0.3768513 - 0.2784451 * x[0], 0.02)
    su = maximum(0.4601818 - 0.3852638 * x[0], 0.02)
    w = sigmoid(-0.1643845) * sigmoid((x[0] - 0.5256900) / 0.005)
    return Mixture([1 - w, w], [Normal(ml, sl), Normal(mu, su)])
