from adr.dist import Normal, Mixture, sin, exp, sigmoid


def model(x):
    t = x[0]
    m1 = 0.5796 * sin(7.7985 * t + -3.5787) + -0.2536
    m2 = -2.5943 + 5.8747 * t + -0.7843 * t * t
    s2 = exp(-1.1826 + -0.9498 * t)
    w = sigmoid(4.4795 * (t - 0.8437))
    return Mixture([1 - w, w], [Normal(m1, 0.2303), Normal(m2, s2)])
