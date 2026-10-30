from adr.dist import Normal, Mixture, sin, exp, sigmoid


def model(x):
    t = x[0]
    m1 = 0.8634 * sin(5.5663 * t + 3.7342) + 0.3269 - 1.4459 * t
    s1 = exp(-0.931 - 1.24 * t)
    m2 = -2.1474 + 4.7674 * t
    s2 = exp(-0.3248 - 2.0116 * t)
    w2 = 0.4733 * sigmoid(50.0 * (t - 0.3288))
    return Mixture([1 - w2, w2], [Normal(m1, s1), Normal(m2, s2)])
