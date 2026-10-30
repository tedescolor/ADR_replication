from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    u = x[0] - 0.5
    m1 = -0.0177 + u * (4.9892 + u * (5.228 + u * (-10.1901 + u * -0.3134)))
    m2 = -0.1002 + u * (2.103 + u * (-3.2118 + u * (-8.9761 + u * -0.5269)))
    s1 = exp(-1.2785 + u * (-1.3041 + u * -1.8968))
    s2 = exp(-2.0256 + u * (2.5899 + u * -6.9796))
    w = sigmoid(0.994 + u * (-5.7074 + u * (2.7005 + u * 8.6297)))
    return Mixture([w, 1 - w], [Normal(m1, s1), Normal(m2, s2)])
