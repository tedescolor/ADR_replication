from adr.dist import Normal, Mixture, exp, log, where


def model(x):
    t = x[0]
    m = 0.273197707 + t * (-12.0696055726 + t * (37.7606615106 - 28.3732122721 * t))
    s = log(1.0 + exp(20.5838 * (t - 0.5455502093))) / 20.5838
    up = m + s * (-1.1081997995 + 25.8359895447 * s)
    lo = m + s * (-6.3591088593 + 21.123979355 * s)
    sd = exp(-0.8881766545 - 1.4055686761 * t)
    w = where(t > 0.5455502093, 0.5, 0.0)
    return Mixture([w, 1.0 - w], [Normal(up, sd), Normal(lo, sd)])
