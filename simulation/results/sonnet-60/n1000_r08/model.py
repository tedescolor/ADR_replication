from adr.dist import SkewNormal, exp


def model(x):
    t = x[0]
    loc = -0.3803 + t * (-6.3012 + t * (20.0561 + t * (-15.3243 + t * 0.5041)))
    scale = exp(-0.5949 + t * (-2.4432 + t * 4.2667))
    shape = 2.5001 + t * (-14.9076 + t * 32.5661)
    return SkewNormal(loc, scale, shape)
