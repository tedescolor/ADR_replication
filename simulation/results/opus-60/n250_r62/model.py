from adr.dist import Mixture, Normal, exp


def model(x):
    t = x[0]
    w = 0.4513
    m1 = -0.0267 + t * (-5.6697 + t * (17.3026 - 12.679 * t))
    m2 = -0.1775 + t * (-7.0226 + t * (19.6138 - 9.8618 * t))
    s1 = exp(-1.1699 - 1.0147 * t)
    s2 = exp(-0.652 - 1.5978 * t)
    return Mixture([w, 1.0 - w], [Normal(m1, s1), Normal(m2, s2)])
