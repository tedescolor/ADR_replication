from adr.dist import StudentT, exp, minimum, maximum


def model(x):
    x0, x1, x2, x3, x4, x5 = x[0], x[1], x[2], x[3], x[4], x[5]
    mu = (3.4263 - 0.0207 * x0 + 2.082 * x1 - 1.662 * x2 + 0.1784 * x4 - 0.2797 * x5
          + 0.5885 * minimum(x3, 1.0) - 0.3395 * minimum(x4, 1.0)
          - 0.421 * x0 * x1 + 0.2207 * x0 * x2 + 0.2397 * x0 * x5 + 0.4067 * x1 * x2)
    ls = (0.1486 - 0.0315 * x0 + 0.2947 * x1 - 0.2156 * x2 + 0.0909 * x3 + 0.1334 * x4
          - 0.8978 * x5 + 0.0183 * x5 * x5 - 0.0268 * x0 * x0 + 0.2071 * x1 * x1
          + 0.2102 * x0 * x5)
    ls = minimum(maximum(ls, -1.5), 1.8)
    return StudentT(exp(1.7111), mu, exp(ls))
