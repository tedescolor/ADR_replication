from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = ((-9.03902 * t + 18.32393) * t + -6.67225) * t + -0.07476
    s1 = exp(-1.01337 + -1.35974 * t)
    m2 = -4.10677 + 13.08561 * t + -9.93873 * t * t
    s2 = exp(-0.29603 + -1.89266 * t)
    w = sigmoid((t - 0.31561) * 20.7834) * 0.52146
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
