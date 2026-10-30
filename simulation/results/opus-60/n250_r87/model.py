from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = 2 * x[0] - 1
    m1 = (((1.1895 * t - 1.9994) * t + 0.5099) * t + 2.8659) * t + 0.0384
    m2 = (((0.8644 * t - 1.7868) * t - 1.1629) * t + 1.2282) * t - 0.062
    s1 = exp(-1.0903 * t - 1.4676)
    s2 = exp(-0.4578 * t - 1.5434)
    w = sigmoid(1.5657 * t - 0.9166)
    return Mixture([w, 1 - w], [Normal(m1, s1), Normal(m2, s2)])
