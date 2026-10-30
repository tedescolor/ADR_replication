from adr.dist import Normal, Mixture, sigmoid, exp


def model(x):
    t = x[0]
    m1 = ((((10.0234 * t + -35.7556) * t + 34.7665) * t + -10.1303) * t + 0.0818)
    s1 = exp(-0.9372 + -1.3098 * t)
    m2 = -2.3635 + 5.0289 * t
    s2 = exp(-1.3229 + -0.9114 * t)
    w = sigmoid(-2.7856 + 3.4537 * t)
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
