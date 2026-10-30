from adr.dist import Mixture, SkewNormal, LogNormal


def model(x):
    """Unconditional: right-skewed noise bump near zero + lognormal heavy tail."""
    return Mixture([0.39164, 0.60836],
                   [SkewNormal(-0.71514, 0.87974, 2.07162), LogNormal(1.45489, 1.82083)])
