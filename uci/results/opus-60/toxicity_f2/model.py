from adr.dist import StudentT, exp, sqrt, maximum, where


def model(x):
    x0, x1, x2, x3, x4, x5 = x[0], x[1], x[2], x[3], x[4], x[5]
    a = x0 - 2.9
    b = x1 - 0.6
    c = x5 - 2.1
    mu = (1.6256 + 0.5479 * x0 + 1.3537 * x1 - 0.8779 * x2 - 0.3660 * x3 + 0.1856 * x4 + 0.3771 * x5
          + 0.2384 * a * c - 0.3867 * a * b + 0.9543 * sqrt(maximum(x3, 0.0)) + 0.5107 * b * b
          - 0.3480 * where(x4 > 0.5, 1.0, 0.0))
    ls = (-0.9743 + 0.3214 * x0 + 0.6051 * x1 - 0.3267 * x2 + 0.1381 * x3 + 0.0869 * x4 - 0.2858 * x5
          + 0.2559 * a * c + 0.1570 * b * c)
    return StudentT(5.7505, mu, exp(ls))
