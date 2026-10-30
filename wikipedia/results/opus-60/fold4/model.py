from adr.dist import LogNormal, SkewNormal, Sequential, log, exp


def model(x):
    f0 = log(x[0])
    f1 = log(1 + x[1])
    f2 = log(1 + x[2])
    f3 = x[3]
    f4 = (x[4] - 14) / 10
    f5 = log(x[5])
    f6 = f4 * f4
    one = f0 < 0.5
    mu = 0.81663 + 0.09497 * f0 + 0.01065 * f1 + 0.03914 * f2 + 0.03964 * f3 + 0.15526 * f4 + 0.17194 * f5 + 0.17672 * f6 + 0.1787 * one
    s = exp(-1.16887 + -0.08205 * f0 + -0.10045 * f3 + -0.15469 * f4)
    m1 = 0.04871 + 0.09124 * f0 + 0.05082 * f1 + 0.00467 * f2 + 0.02841 * f3 + -0.02188 * f4 + 0.11062 * f5 + 0.18718 * f6

    def cond(y0):
        loc = m1 + 0.29938 * y0 + -0.74005 * log(y0)
        scale = exp(-0.55993 + 0.07848 * f5 + 0.0429 * y0)
        return SkewNormal(loc, scale, 1.32062)

    return Sequential(LogNormal(mu, s), cond)
