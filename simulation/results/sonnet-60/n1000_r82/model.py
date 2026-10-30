from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    u = x[0] - 0.5
    m1 = -0.086 + u * (2.8 + u * (-3.935 + u * (-16.072 + u * 10.509)))
    s1 = exp(-1.367 + u * (-1.402 + u * -1.139))
    m2 = 0.188 + 4.841 * u
    s2 = exp(-1.873)
    p = sigmoid(-1.504 + 5.364 * u)
    return Mixture([1 - p, p], [Normal(m1, s1), Normal(m2, s2)])
