from adr.dist import Normal, Mixture, exp, sigmoid, maximum


def model(x):
    t = x[0]
    base = ((((((-59.76611) * t + 192.696651) * t + -226.260321) * t + 100.685989) * t + -2.026192) * t + -6.353899) * t + 0.033026
    r = maximum(t - 0.370369, 0.0)
    sd = exp(((-0.832819) * t + -0.519261) * t + -0.97866)
    w = sigmoid(-0.011996)
    return Mixture([w, 1.0 - w], [Normal(base + 8.891043 * r * r, sd), Normal(base, sd)])
