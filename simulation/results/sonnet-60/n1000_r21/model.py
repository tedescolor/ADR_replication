from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = ((((3.627)*t + -18.157)*t + 26.338)*t + -9.4224)*t + 0.14761
    m2 = (((-9.3285)*t + 11.072)*t + -2.2973)*t + -0.46124
    s1 = exp((-1.3619)*t + -0.95809)
    s2 = exp((-1.5005)*t + -0.79934)
    w = sigmoid(1.7767 * (t - 0.82618))
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
