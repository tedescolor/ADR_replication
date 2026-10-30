from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = -4.380300 + t * (22.471307 + t * (-45.802121 + t * (46.492515 + t * -19.788736)))
    m2 = 0.032818 + t * (-7.744105 + t * (20.199406 + t * -9.942570))
    s1 = exp(-0.392952 - 1.961218 * t)
    s2 = exp(-0.793614 - 1.444599 * t)
    w = sigmoid(-7.726388 + t * (19.095742 + t * -11.480788))
    return Mixture([w, 1 - w], [Normal(m1, s1), Normal(m2, s2)])
