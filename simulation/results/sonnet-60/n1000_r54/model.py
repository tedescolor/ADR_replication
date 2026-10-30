from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = ((((6.8617)*t+-28.316)*t+28.998)*t+-8.5431)*t+-0.0047602
    m2 = (((-10.51)*t+21.366)*t+-8.837)*t+0.48121
    s1 = exp(-0.76179 + -1.4799 * t)
    s2 = exp(-0.26254 + -2.1966 * t)
    w = sigmoid(-5.6354 + (14.81 + -9.6169 * t) * t)
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
