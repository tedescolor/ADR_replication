from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = 0.13652382 + t * (-10.22946214 + t * (36.91752257 + t * (-41.02386262 + t * 13.25765602)))
    s1 = exp(-1.05910758 + t * (-0.22518317 - 1.36487441 * t))
    m2 = -3.84573064 + t * (8.98507617 - 2.5716502 * t)
    s2 = exp(0.30429584 - 2.83613425 * t)
    w = sigmoid(-4.79199137 + 5.84664189 * t)
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
