from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = 0.2008 + t * (-10.8408 + t * (34.88432 + t * (-33.71495 + t * 8.41781)))
    m2 = -2.35908 + 4.93912 * t
    s1 = exp(-0.86109 + -1.54485 * t)
    s2 = exp(-0.61786 + -1.40099 * t)
    w = sigmoid(6.63852 + t * (-17.891 + t * 11.62254))
    return Mixture([w, 1 - w], [Normal(m1, s1), Normal(m2, s2)])
