from adr.dist import Normal, Mixture, exp, maximum, sigmoid


def model(x):
    t = x[0]
    # lower branch: quartic in t; upper branch splits off at a hinge
    m1 = -0.115341 + t * (-6.078842 + t * (21.808462 + t * (-21.064526 + t * 4.453847)))
    h = maximum(t - 0.101472, 0.0)
    m2 = m1 + h * (-3.099362 + 7.84345 * h)
    s = exp(-0.819242 + -1.404767 * t)
    w = sigmoid(-0.100349)
    return Mixture([1.0 - w, w], [Normal(m1, s), Normal(m2, s)])
