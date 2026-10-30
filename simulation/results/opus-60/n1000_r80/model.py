from adr.dist import Normal, Mixture, exp, log, sigmoid


def model(x):
    t = 2 * x[0] - 1
    f = ((((0.4357 * t + 0.487) * t - 2.4138) * t - 0.7882) * t + 1.4996) * t - 0.1824
    s = 0.1162
    z = (x[0] - 0.6818) / s
    up = f + 10.7171 * s * (log(1 + exp(-abs(z))) + (z + abs(z)) / 2)
    sd = exp((-0.1927 * t - 0.6627) * t - 1.4351)
    w = sigmoid(0.0894)
    return Mixture([w, 1 - w], [Normal(up, sd), Normal(f, sd)])
