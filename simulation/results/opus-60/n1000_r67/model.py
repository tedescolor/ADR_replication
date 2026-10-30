from adr.dist import Normal, Mixture, exp, sigmoid, maximum


def model(x):
    """Trunk curve m1 that splits at x0 = 0.43937 into a second, rising branch m2."""
    z = 2 * x[0] - 1
    r = maximum(x[0] - 0.43937, 0.0)
    m1 = (-0.049323 + z * (1.591902 + z * (-2.476932 + z * (-3.749209 + z * (7.384672 + z * (3.680542 + z * (-10.290576 + z * (-2.055526 + z * 4.952326))))))))
    m2 = m1 + r * (0.708873 + r * (13.186635 + r * -5.931701))
    s1 = exp((-1.415834 + z * (-0.60207 + z * -0.26391)))
    s2 = exp((-1.986528 + z * (1.56269 + z * -2.110975)))
    w = sigmoid(-0.106931) * sigmoid((x[0] - 0.43937) * 100.0)
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
