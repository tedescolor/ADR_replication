from adr.dist import StudentT, exp, abs, maximum, minimum


def model(x):
    a0 = x[0] - 2.901
    a1 = x[1] - 0.623
    a2 = x[2] - 1.285
    a3 = x[3] - 0.231
    a4 = x[4] - 0.457
    a5 = x[5] - 2.118
    mu = (3.8509 + 0.5072 * a0 + 1.3655 * a1 - 0.6514 * a2 + 0.5201 * a3 - 0.0454 * a4 + 0.4007 * a5
          + 0.245 * a0 * a5 - 0.1972 * a3 * a5 + 0.5534 * a2 * a4 + 0.5808 * a1 * a1)
    ls = (-0.7092 + 0.3219 * a0 + 0.5706 * a1 - 0.2369 * a2 + 0.0264 * a3 + 0.1117 * a4 - 0.2248 * a5
          + 0.1148 * abs(a5) + 0.2014 * a0 * a5 - 0.5153 * a1 * a3)
    ls = minimum(maximum(ls, -2.0), 1.0)
    return StudentT(exp(1.6439), mu, exp(ls))
