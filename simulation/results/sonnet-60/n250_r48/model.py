from adr.dist import Normal, Mixture, sigmoid, exp


def model(x):
    t = x[0]
    ml = 0.15558 + t * (-10.70842 + t * (36.93785 + t * (-38.67251 + t * 11.29357)))
    mu = -1.68939 + t * 4.24864
    p = sigmoid(-4.67836 + 5.38201 * t)
    return Mixture([p, 1 - p], [Normal(mu, exp(0.24273 + -2.5625 * t)),
                                Normal(ml, exp(-0.88903 + -1.30948 * t))])
