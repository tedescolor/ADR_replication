from adr.dist import SkewNormal, exp


def model(x):
    t = x[0]
    m = -0.29133245 + t * (-4.07513033 + t * (14.09881686 + t * -6.8485217))
    s = exp(-0.9166981 + t * (-1.02108534 + t * 2.90782683))
    return SkewNormal(m, s, 1.71606835 - 12.95973668 * t)
