from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    ma = ((((((-112.61144)*t+351.00103)*t+-405.87651)*t+197.29646)*t+-27.79698)*t+-2.91946)*t+-0.14348
    sa = exp(((-1.13896)*t+-0.09302)*t+-1.14047)
    mb = (4.80359)*t+-2.19624
    sb = exp((-1.34127)*t+-0.84347)
    pb = sigmoid(4.3142 * (t - 0.8471))
    return Mixture([1 - pb, pb], [Normal(ma, sa), Normal(mb, sb)])
