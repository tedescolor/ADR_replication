from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    """Two-branch normal mixture: quartic means, log-quadratic scales, logit-linear weight."""
    u = 2 * x[0] - 1
    w = sigmoid((-0.457067)*u+0.333406)
    m1 = ((((0.0145903)*u+-1.33432)*u+1.37077)*u+2.44708)*u+0.00947308
    m2 = ((((1.11853)*u+-1.88843)*u+-1.66544)*u+1.55986)*u+-0.073948
    s1 = exp(((-0.495169)*u+-0.582223)*u+-1.41938)
    s2 = exp(((-0.744632)*u+-0.380441)*u+-1.28584)
    return Mixture([w, 1 - w], [Normal(m1, s1), Normal(m2, s2)])
