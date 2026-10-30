from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = ((((11.717*t+-38.1116)*t+35.2263)*t+-9.8659)*t+0.0736)
    m2 = ((-2.42*t+8.8286)*t+-3.843)
    s1 = exp(-0.8516 + -1.3051 * t)
    s2 = exp(-0.6594 + -1.6644 * t)
    w = sigmoid(-3.6596 + 4.8416 * t)
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
