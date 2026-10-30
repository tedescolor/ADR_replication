from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = ((-11.4008 * t + 14.7407) * t - 4.0783) * t - 0.3384
    m2 = -2.4203 + 5.0728 * t
    s = exp(-0.9593 - 1.2614 * t)
    w = 0.5 * sigmoid(2000.0 * (t - 0.2098))
    return Mixture([1.0 - w, w], [Normal(m1, s), Normal(m2, s)])
