from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    t = x[0] - 0.5
    z = ((-17.88)*t+9.541)*t+-0.8608
    ma = ((((10.28)*t+-14.84)*t+-3.82)*t+2.667)*t+-0.131
    mb = ((-2.015)*t+6.143)*t+-0.006327
    sa = exp((-1.348)*t+-1.504)
    sb = exp((-1.928)*t+-1.402)
    return Mixture([sigmoid(-z), sigmoid(z)], [Normal(ma, sa), Normal(mb, sb)])
