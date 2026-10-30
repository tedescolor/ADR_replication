from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    mA = -0.18895 + t * (-5.7968 + t * (18.189 + t * -13.301))
    sA = exp(-0.88867 + -1.2361 * t)
    mB = -1.9864 + t * (4.0014 + 0.61398 * t)
    w = sigmoid(-2.9052 + 3.4327 * t)
    return Mixture([1 - w, w], [Normal(mA, sA), Normal(mB, 0.16272)])
