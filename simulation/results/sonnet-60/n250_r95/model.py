from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = 0.04224355 + t * (-9.66715449 + t * (32.9139717 + t * (-32.50755125 + t * 8.20243389)))
    s1 = exp(-0.79868111 - 1.47542954 * t)
    m2 = -2.44717406 + 5.05036081 * t
    w = sigmoid(-3.66875552 + 4.77420604 * t)
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, exp(-2.13619948))])
