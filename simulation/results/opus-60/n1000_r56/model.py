from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0] - 0.5
    w = sigmoid(-0.0503136) * sigmoid(1000000.0 * (t - 0.0664621))
    upper = Normal((((((1.83474) * t + 2.68309) * t + -11.0945) * t + 4.43385) * t + 5.13232) * t + 0.0115188, exp(((-0.790752) * t + -1.37106) * t + -1.42056))
    lower = Normal((((-7.29348) * t + -6.17868) * t + 2.98361) * t + -0.12013, exp((-1.3837) * t + -1.55228))
    return Mixture([1.0 - w, w], [upper, lower])
