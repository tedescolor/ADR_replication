from adr.dist import SkewNormal, exp


def model(x):
    t = x[0]
    mu = (((10.38516374 * t - 27.06535651) * t + 28.90893504) * t - 9.46775377) * t + 0.49650507
    s = exp((5.41817998 * t - 4.07666228) * t - 0.18912748)
    return SkewNormal(mu, s, -3.56199341)
