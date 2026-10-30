from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    t = x[0]
    w = -13.894 + t * (35.166 - 21.722 * t)
    m1 = 0.027 + t * (-7.534 + t * (19.837 - 9.778 * t))
    m2 = -5.172 + t * (15.51 - 11.308 * t)
    return Mixture(
        [sigmoid(-w), sigmoid(w)],
        [Normal(m1, exp(-0.921 - 1.194 * t)), Normal(m2, exp(0.042 - 2.553 * t))],
    )
