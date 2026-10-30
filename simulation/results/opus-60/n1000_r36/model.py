from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = 2 * x[0] - 1
    f = 0.47667 + t * (3.67601 + t * (1.93723 + t * (-1.99347 + t * (-0.75498))))
    h = sigmoid(3.51865 * (t - 0.31541))
    m1 = f + h * (-1.65403 + t * (-0.95603 + t * (1.65283)))
    m2 = f + h * (-2.46892 + t * (-2.35696 + t * (0.02865)))
    s1 = exp(-1.41587 + t * (-0.81045 + t * (-0.35095)))
    s2 = exp(-1.46716 + t * (-0.6798 + t * (-0.09738)))
    w = sigmoid(0.23439 + -0.10742 * t)
    return Mixture([w, 1 - w], [Normal(m1, s1), Normal(m2, s2)])
