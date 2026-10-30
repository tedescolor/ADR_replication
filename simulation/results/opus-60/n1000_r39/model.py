from adr.dist import Normal, StudentT, Mixture, exp, sigmoid


def model(x):
    x = x[0]
    m1 = -0.12256 + x * (-9.22088 + x * (28.82455 + x * (-22.8321 + x * (5.88418))))
    m2 = 0.38721 + x * (-10.70169 + x * (34.91148 + x * (-36.09667 + x * (10.54148))))
    s1 = exp(-1.28404 + x * (0.11328 + x * (-1.10787)))
    s2 = exp(-1.66866 + x * (2.00112 + x * (-2.81231)))
    w = sigmoid(0.13652)
    return Mixture([w, 1 - w], [StudentT(21.737, m1, s1), Normal(m2, s2)])
