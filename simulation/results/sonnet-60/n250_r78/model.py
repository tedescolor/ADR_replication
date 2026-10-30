from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    t = x[0] - 0.5
    u = x[0] - 0.6
    ml = ((((11.4447)*t+-15.3047)*t+-4.0795)*t+2.8332)*t+-0.1228
    mu = ((-2.2241)*t+6.2834)*t+-0.0132
    w = sigmoid(0.2962 + 4.5926 * u + -15.8935 * u * u)
    s1 = exp(-1.4504 + -1.1769 * t)
    s2 = exp(-1.6266 + -1.6159 * t)
    return Mixture([1 - w, w], [Normal(ml, s1), Normal(mu, s2)])
