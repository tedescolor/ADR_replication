from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    x0 = x[0]
    ma = ((((((-122.12795) * x0 + 426.68719) * x0 + -561.63158) * x0 + 329.41231) * x0 + -76.59924) * x0 + 3.44218) * x0 + -0.19498
    mb = ((1.51876) * x0 + 2.47465) * x0 + -1.33507
    sa = exp(-0.94367 + -1.17439 * x0)
    wb = sigmoid(-2.41449 + 2.70402 * x0)
    return Mixture([1 - wb, wb], [Normal(ma, sa), Normal(mb, 0.15828)])
