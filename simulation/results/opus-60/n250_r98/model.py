from adr.dist import Normal, Mixture, sin, exp, sigmoid


def model(x):
    t = x[0]
    m1 = -0.1953 - 0.3621 * t + 0.659 * sin(6.8467 * t + 3.1136)
    s1 = exp(-1.4118 * t - 0.9801)
    m2 = ((-7.38 * t + 15.8315) * t - 6.1497) * t + 0.2346
    s2 = exp(-1.1924 * t - 0.9159)
    q = sigmoid(1.414 * t - 1.2885)
    return Mixture([1 - q, q], [Normal(m1, s1), Normal(m2, s2)])
