from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    u = x[0] - 0.5
    ma = ((((12.0917)*u+(-16.1308))*u+(-4.1442))*u+(2.914))*u+(-0.1353)
    mb = (((-11.134)*u+(5.5325))*u+(5.0997))*u+(-0.0394)
    sa = exp(((-2.447)*u+(-1.0396))*u+(-1.3775))
    sb = exp((-1.6777)*u+(-1.4659))
    p = sigmoid(-0.1117 + 0.4502 * u)
    return Mixture([1 - p, p], [Normal(ma, sa), Normal(mb, sb)])
