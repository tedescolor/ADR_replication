from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    t = x[0]
    m1 = -0.3617 + t * (-8.152 + t * (30.78 + t * (-32.19 + t * 8.922)))
    m2 = 0.08606 + t * (-7.568 + t * (19.94 + t * -9.991))
    s1 = exp(-1.938 + t * (1.898 + t * -2.44))
    s2 = exp(-0.8737 - 1.345 * t)
    w = sigmoid(1.238 - 1.265 * t)
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
