from adr.dist import Normal, StudentT, arctan2, sin, cos, sqrt, maximum, where, pi


def saw(x, p, off):
    t = 2 * pi * (x - off) / p
    fr = arctan2(sin(t), cos(t)) / (2 * pi)
    return fr, (x - off) / p - fr


def model(x):
    f0, b0 = saw(x[0], 0.5, 3.3)
    f2, b2 = saw(x[2], 0.05, 0.02)
    s2 = maximum(b2 - 1, 0)
    g2 = where(b2 >= 2, f2, 0)
    e = where(b0 < -0.5, 1, 0)
    g = (1.27168 + 0.0233745 * b0 + -0.098816 * f0 + 0.110341 * s2 + -0.0226632 * g2
         + 0.0113262 * b0 * s2 + 0.0168464 * f0 * s2 + -0.0029168 * b0 * g2
         + 0.009436 * f0 * g2 + 1.38637e-05 * b0 * b0 + -0.00287955 * g2 * s2
         + -0.00546107 * f0 * b0 + -0.00762326 * s2 * s2 + 0.0307745 * e
         + -0.00075123 * b0 * s2 * s2 + -0.0210956 * sqrt(s2))
    if x[3] > 20:
        return Normal(1.08979 + 0.000923393 * x[3] + 0.0765979 * g, 0.00272927)
    return StudentT(4, g, 0.0071106)
