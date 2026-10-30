from adr.dist import Normal, Mixture, sigmoid, exp, sin


def model(x):
    x0 = x[0]
    m1 = 0.371229 + -1.445717 * x0 + 0.860371 * sin(5.635075 * x0 + -2.636436)
    s1 = exp((-1.272527) * x0 + -0.909814)
    m2 = 5.239172 * x0 + -2.540937
    s2 = exp(-1.950163)
    pu = sigmoid(100.0 * (x0 - 0.407602)) * sigmoid(0.046582)
    return Mixture([1.0 - pu, pu], [Normal(m1, s1), Normal(m2, s2)])
