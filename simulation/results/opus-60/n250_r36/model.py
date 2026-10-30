from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = 0.161 + t * (-8.719 + t * (22.139 - 11.095 * t))
    s1 = exp(-1.112 + t * (0.193 - 1.281 * t))
    m2 = 3.568 - 4.459 * t
    s2 = exp(2.446 - 4.823 * t)
    w = 0.4973 * sigmoid(3654.0 * (t - 0.596))
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
