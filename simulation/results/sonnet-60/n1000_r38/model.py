from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    u = x[0] - 0.5
    m1 = (((((2.3653)*u+-3.0545)*u+-10.7997)*u+5.7545)*u+5.0687)*u+-0.0265
    m2 = ((-9.0633)*u+2.6138)*u+0.0307
    s1 = exp(-0.9639 + -1.1318 * x[0])
    s2 = exp(-0.0314 + -2.2344 * x[0])
    p = sigmoid(-8.1697 + 20.4172 * x[0] + -12.4272 * x[0] * x[0])
    return Mixture([1 - p, p], [Normal(m1, s1), Normal(m2, s2)])
