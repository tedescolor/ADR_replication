from adr.dist import Normal, Mixture, sigmoid, exp


def model(x):
    t = x[0]
    m1 = -0.0018 + t * (-9.3883 + t * (32.2099 + t * (-31.8567 + t * 7.9305)))
    m2 = -2.3523 + 5.0049 * t
    s1 = exp(-0.9535 - 1.341 * t)
    s2 = exp(-1.9042)
    w = sigmoid(-4.0067 + 5.14 * t)
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
