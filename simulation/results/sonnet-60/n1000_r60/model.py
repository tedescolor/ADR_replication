from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    t = x[0]
    m1 = 0.036158 + t * (-7.2788 + t * (19.241 - 9.4801 * t))
    m2 = -3.0843 + t * (8.4873 + t * (-3.3361 - 3.0534 * t))
    s1 = exp(-0.8688 - 1.4012 * t)
    s2 = exp(-0.33489 - 1.9651 * t)
    z = 15.14 + t * (-53.602 + t * (37.643 + t * (36.873 - 36.81 * t)))
    p = sigmoid(z)
    return Mixture([p, 1 - p], [Normal(m1, s1), Normal(m2, s2)])
