from adr.dist import Normal, Mixture, exp, maximum, where


def model(x):
    t = x[0]
    m1 = (((13.56461 * t - 40.72895) * t + 36.0297) * t - 9.82409) * t + 0.04766
    m2 = m1 + 8.25942 * maximum(t - 0.54528, 1e-12) ** 1.12844
    s1 = exp(-0.79318 - 1.57741 * t)
    s2 = exp(0.44007 - 2.69936 * t)
    w = where(t > 0.54528, 0.57088, 0.0)
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
