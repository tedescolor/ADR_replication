from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    u = x[0] - 0.5
    ma = (((((6.539)*u + 7.911)*u + -19.17)*u + -2.747)*u + 3.185)*u + -0.2465
    mb = (((-8.24)*u + 4.357)*u + 4.739)*u + 0.07775
    wb = sigmoid(0.2615 + -0.6097 * u)
    return Mixture([1 - wb, wb], [Normal(ma, exp(-1.575 + -1.311 * u)), Normal(mb, exp(-1.51 + -1.282 * u))])
