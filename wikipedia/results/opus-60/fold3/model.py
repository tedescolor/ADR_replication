from adr.dist import LogNormal, SkewNormal, Sequential, Mixture, log, exp, sigmoid


def model(x):
    f0 = log(x[0])
    f1 = log(1 + x[1])
    f2 = log(1 + x[2])
    f5 = log(x[5])
    w = sigmoid(0.09372 + 0.2564*f0 + 0.1245*f1 + 0.5154*f2 - 0.5173*x[3] - 0.04927*x[4] - 1.01*f5)
    mu = 0.825 + 0.07635*f0 + 0.00518*f1 + 0.07385*f2 - 0.03422*x[3] - 0.02889*x[4] + 0.1054*f5 + 0.001474*x[4] * x[4]
    sd = exp(-0.8709 - 0.15*f0 - 0.09329*f1 - 0.0581*f2 - 0.001013*x[3] - 0.02433*x[4] + 0.1652*f5)

    def cond(y, d, e, k):
        loc = 0.188 + 0.09043*f0 + 0.05558*f1 - 0.03165*f2 + 0.03317*x[3] - 0.004862*x[4] + 0.1134*f5 + 0.1143*y + 0.0117*y * y
        return SkewNormal(loc + d, exp(-0.4161 - 0.08195*f0 - 0.06138*f1 - 0.05508*f2 + 0.03711*x[3] - 0.007829*x[4] + 0.1819*f5 + 0.07783*y + e), k)

    a = Sequential(LogNormal(mu, sd), lambda y: cond(y, 0, 0, 0.4932))
    b = Sequential(LogNormal(mu + 0.4115, sd * 1.0),
                   lambda y: cond(y, -0.3205, -0.2043, 0.4932))
    return Mixture([w, 1 - w], [a, b])
