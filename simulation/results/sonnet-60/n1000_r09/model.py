from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    p = 0.4929 * sigmoid(97.35 * (t - 0.2521))
    m1 = (((((-7.937*t+31.07)*t+-54.17)*t+40.42)*t+-10.47)*t+0.1051)
    m2 = (((-7.204*t+13.96)*t+-3.322)*t+-0.9131)
    s1 = exp(((-0.7946*t+-0.6405)*t+-0.8927))
    s2 = exp(((-3.926*t+3.569)*t+-2.195))
    return Mixture([1 - p, p], [Normal(m1, s1), Normal(m2, s2)])
