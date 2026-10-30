from adr.dist import Normal, Mixture, exp, log


def model(x):
    u = x[0] - 0.5
    ma = (((-9.97042) * u + 4.96559) * u + 5.02468) * u + -0.00382734
    r = log(1 + exp((x[0] - 0.520229) * 29.604)) / 29.604
    mb = ma - (((6.16019) * r + 4.2859) * r + 0)
    sa = exp(((-1.51665) * u + -0.967062) * u + -1.42607)
    sb = exp((-1.57505) * u + -1.51584)
    w = 1 / (1 + exp(0.0813836))
    return Mixture([w, 1 - w], [Normal(ma, sa), Normal(mb, sb)])
