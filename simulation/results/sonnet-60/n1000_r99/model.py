from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0] - 0.5
    u = (x[0] - 0.75) * 5
    mA = ((((((-15.06)*t+-20.55)*t+21.63)*t+-8.433)*t+-6.113)*t+2.284)*t+-0.02661
    sA = exp((-1.427)*t+-1.573)
    mB = (((-10.71)*t+5.284)*t+5.135)*t+-0.04775
    sB = exp((-1.148)*t+-1.493)
    p = sigmoid((-0.2822)*u+0.003104)
    return Mixture([1 - p, p], [Normal(mA, sA), Normal(mB, sB)])
