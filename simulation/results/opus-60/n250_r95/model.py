from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    """Two-branch Gaussian mixture: main cubic curve with log-linear sd, plus a lower
    cubic branch that switches on (weight ~0.45) for x > ~0.55."""
    x = x[0]
    m1 = ((-9.73833 * x + 19.65144) * x - 7.36805) * x - 0.04083
    s1 = exp(-1.74745 * x - 0.76743)
    m2 = ((-27.02611 * x + 48.52477) * x - 27.40828) * x + 4.79028
    s2 = exp(-2.47011)
    w = sigmoid(15.25471 * x - 8.2448) * sigmoid(-0.18089)
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
