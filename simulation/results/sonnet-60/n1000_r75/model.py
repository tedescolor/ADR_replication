from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    pb = sigmoid(1.252 + -1.532 * t)
    ma = ((((10.69)*t + -36.84)*t + 35.41)*t + -10.33)*t + 0.08259
    mb = (((-8.856)*t + 18.12)*t + -6.672)*t + -0.06823
    sa = exp((-1.756)*t + -0.5977)
    sb = exp((-1.219)*t + -0.9128)
    return Mixture([1 - pb, pb], [Normal(ma, sa), Normal(mb, sb)])
