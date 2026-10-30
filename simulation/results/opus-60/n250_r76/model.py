from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    u = x[0] - 0.5
    g = sigmoid((x[0] - 0.6126) * 200.0)
    mu1 = ((-9.2291 * u + 5.2483) * u + 4.9049) * u - 0.0341
    mu2 = (-11.288 * u + 4.5453) * u - 0.3662
    sd1 = exp(-1.3912 - 0.7714 * u - 0.2568 * g)
    sd2 = exp(-2.1011)
    p2 = 0.5 * g
    return Mixture([1.0 - p2, p2], [Normal(mu1, sd1), Normal(mu2, sd2)])
