from adr.dist import Normal, Mixture, sigmoid, sin, exp


def model(x):
    t = x[0]
    wb = sigmoid(5.957 * t - 4.467)
    ma = -0.294 + 0.517 * sin(7.713 * t - 3.423)
    sa = exp(-1.018 - 0.823 * t)
    return Mixture([1 - wb, wb], [Normal(ma, sa), Normal(4.773 * t - 2.184, 0.135)])
