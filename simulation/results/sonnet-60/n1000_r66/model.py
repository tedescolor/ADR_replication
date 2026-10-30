from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    u = x[0]
    m1 = 0.115519 + u * (-10.6249 + u * (38.1232 + u * (-42.0951 + u * 13.5215)))
    m2 = -3.60219 + u * (8.28146 + u * -2.14246)
    s1 = exp(-0.96747 + u * (-1.11039 + u * -0.141987))
    s2 = exp(-0.564597 + u * -1.7027)
    w = sigmoid(-3.55329 + u * 4.29809)
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
