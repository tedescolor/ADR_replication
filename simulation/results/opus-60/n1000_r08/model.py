from adr.dist import Normal, Mixture, exp, maximum, sigmoid


def model(x):
    """Two-branch Gaussian mixture: shared trunk, branches split after x = c."""
    z = 2 * x[0] - 1
    r = maximum(x[0] - 0.48825, 0)
    t = 0.05882 + z * (3.67527 + z * (5.51836 + z * (1.72996 + z * (-2.30574 + z * (-0.93265 + z * (1.2707))))))
    m1 = t + r * (-1.7902 + r * (-21.28045))
    m2 = t + r * (-5.02333 + r * (-28.36742))
    ls = -1.42523 + z * (-0.71504 + z * (-0.27374))
    s1 = exp(ls)
    s2 = s1
    w = sigmoid(-0.07204)
    return Mixture([w, 1 - w], [Normal(m1, s1), Normal(m2, s2)])
