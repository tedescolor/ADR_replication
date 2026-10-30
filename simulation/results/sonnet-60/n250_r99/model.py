from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    ml = ((-9.289 * t + 18.9788) * t - 7.0983) * t - 0.0286
    mu = (-7.8822 * t + 9.8274) * t - 2.8519
    sl = exp(-1.0299 - 1.0303 * t)
    su = exp(-0.2655 - 2.1433 * t)
    pu = sigmoid(-3.9086 + 4.931 * t)
    return Mixture([1 - pu, pu], [Normal(ml, sl), Normal(mu, su)])
