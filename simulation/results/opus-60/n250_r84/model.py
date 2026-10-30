from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = 0.25 + t * (-12.006 + t * (48.947 + t * (-83.925 + t * (80.655 + t * -31.469))))
    s1 = exp(-0.886 - 1.517 * t)
    m2 = -5.382 + t * (16.341 - 11.964 * t)
    s2 = exp(-0.252 - 2.347 * t)
    w = sigmoid(5.012 * (t - 0.851))
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
