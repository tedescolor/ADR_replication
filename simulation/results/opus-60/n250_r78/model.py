from adr.dist import Normal, Mixture, where


def model(x):
    t = x[0]
    ml = 0.06699784 + t * (-10.10969248 + t * (34.16270027 + t * (-34.66607766 + t * 9.615342)))
    hi = t > 0.47302207
    sl = where(hi, 0.14564, 0.33514)
    mu = -2.45337315 + 5.10899613 * t
    w = where(hi, 0.576845, 0.0)
    return Mixture([1 - w, w], [Normal(ml, sl), Normal(mu, 0.14029)])
