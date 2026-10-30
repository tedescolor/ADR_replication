from adr.dist import Normal, StudentT, Mixture, exp, sigmoid


def model(x):
    t = 2 * x[0] - 1
    w = sigmoid(4.10037*(x[0]-0.836123))
    a = Normal((((-1.10223)*t+1.13768)*t+2.4419)*t+0.0438751, exp((-0.58989)*t+-1.5175))
    b = Normal(((-2.22839)*t+1.45954)*t+-0.139969, exp(-1.98547))
    return Mixture([1 - w, w], [a, b])
