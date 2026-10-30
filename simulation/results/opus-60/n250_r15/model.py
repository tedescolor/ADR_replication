from adr.dist import Normal, Mixture, sigmoid, exp


def model(x):
    t = x[0]
    m1 = 0.0475 + t * (-9.1176 + t * (27.6757 + t * (-21.9785 + t * 5.9936)))
    s1 = exp(-0.6005 + t * (-4.5282 + t * (8.6113 - 5.9735 * t)))
    m2 = -4.8136 + t * (14.6837 - 10.8484 * t)
    s2 = exp(0.3444 - 2.6835 * t)
    w = sigmoid(-0.0837) * sigmoid(100 * (t - 0.5649))
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
