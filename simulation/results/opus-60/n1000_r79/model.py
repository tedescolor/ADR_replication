from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = 2.0 * x[0] - 1.0
    m1 = -0.102032 + t * (1.26568 + t * (-1.07812 + t * (-1.75386 + t * (0.742904))))
    m2 = 0.027939 + t * (3.07005 + t * (-0.549198))
    s1 = exp(-1.48328 + t * (-0.676314 + t * (-0.229669)))
    s2 = exp(-1.59678 + t * (0.221072 + t * (-1.06109)))
    w = sigmoid(-0.0242178) * sigmoid(3.30236 + 13.7674 * t)
    return Mixture([1.0 - w, w], [Normal(m1, s1), Normal(m2, s2)])
