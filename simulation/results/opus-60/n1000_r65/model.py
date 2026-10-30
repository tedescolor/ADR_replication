from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = 2 * x[0] - 1
    m1 = ((((0.66968)*t+-2.2497)*t+-0.82307)*t+1.6001)*t+-0.24223
    m2 = ((((-0.3491)*t+-0.80327)*t+1.32376)*t+2.22779)*t+0.08288
    s1 = exp((-0.59257)*t+-1.60819)
    s2 = exp((-0.63591)*t+-1.52697)
    w = sigmoid((-0.02199)*t+0.09211)
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
