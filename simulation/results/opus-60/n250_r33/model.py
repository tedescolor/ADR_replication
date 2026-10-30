from adr.dist import StudentT, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    # lower branch: quartic trend, log-linear scale
    m1 = (((10.843613 * t - 36.973823) * t + 34.827549) * t - 9.720304) * t + 0.015583
    s1 = exp(-1.37684 - 0.7545 * t)
    # upper branch appears for x0 > ~0.56: linear trend, constant scale
    m2 = 4.893837 * t - 2.262065
    s2 = exp(-2.02705)
    w = 0.474354 * sigmoid(40.0 * (t - 0.560684))
    return Mixture([1.0 - w, w], [StudentT(10.7757, m1, s1), StudentT(4.66399, m2, s2)])
