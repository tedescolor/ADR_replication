from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = (((0.909 * t + -11.69) * t + 20.78) * t + -7.48) * t + -0.01726
    m2 = (-10.08 * t + 13.63) * t + -4.478
    s1 = exp(-0.9178 + -1.206 * t)
    s2 = exp(-0.8592 + -1.441 * t)
    p2 = sigmoid((-13.61 * t + 22.03) * t + -8.697)
    return Mixture([1 - p2, p2], [Normal(m1, s1), Normal(m2, s2)])
