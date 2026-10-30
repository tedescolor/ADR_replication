from adr.dist import LogNormal, SkewNormal, Sequential, log, exp, where


def model(x):
    f0 = log(x[0])
    f1 = log(x[1] + 1)
    f2 = log(x[2] + 1)
    f3 = x[3]
    f4 = x[4] / 10
    f5 = log(x[5])
    mu0 = 0.9529 + 0.0769*f0 + -0.0016*f1 + 0.0348*f2 + 0.0343*f3 + -0.3173*f4 + 0.2416*f5 + 0.1751*f4*f4 + -0.0192*f5*f5 + where(x[0] < 1.5, 0.1768, 0)
    s0 = exp(-0.95 + -0.0565*f0 + 0.0102*f1 + 0.0332*f2 + -0.1432*f3 + -0.1982*f4 + -0.027*f5)
    m1 = 0.5548 + 0.1529*f0 + 0.0544*f1 + 0.0121*f2 + -0.0218*f3 + -0.0407*f4 + 0.0789*f5
    s1 = -0.4997 + -0.0764*f0 + -0.0108*f1 + 0.0132*f2 + 0.0164*f3 + -0.1112*f4 + 0.1035*f5

    def cond(y0):
        l0 = log(y0)
        return SkewNormal(m1 + -1.0033*l0 + 0.5629*l0*l0, exp(s1 + 0.3172*l0), 1.4782)

    return Sequential(LogNormal(mu0, s0), cond)
