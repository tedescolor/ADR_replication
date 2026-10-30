from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    u = 2 * x[0] - 1
    m1 = (((((-0.83676)*u+0.8615)*u+-0.61386)*u+-1.4536)*u+1.0147)*u+0.0071215
    m2 = (((-1.4472)*u+1.3425)*u+2.6015)*u+-0.03036
    s1 = exp((-0.73744)*u+-1.4872)
    s2 = exp((-0.53913)*u+-1.5466)
    w = sigmoid((-0.73036)*u+0.38581)
    return Mixture([w, 1 - w], [Normal(m2, s2), Normal(m1, s1)])
