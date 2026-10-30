from adr.dist import Normal, Mixture, exp, log, sigmoid


def model(x):
    t = x[0]
    # shared trunk (cubic); for t > ~0.63 a second branch bends away downward
    m1 = 0.00463 - 7.17399 * t + 19.01675 * t * t - 9.33259 * t * t * t
    h = exp(-2.40639)
    z = (t - 0.62882) / h
    g = h * (log(1 + exp(-abs(z))) + (z + abs(z)) / 2)  # stable softplus
    m2 = m1 - 9.17562 * g
    s = exp(-0.83812 - 1.3372 * t)
    w = sigmoid(-0.39766)
    return Mixture([1 - w, w], [Normal(m1, s), Normal(m2, s)])
