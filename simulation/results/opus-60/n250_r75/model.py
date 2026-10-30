from adr.dist import Normal, Mixture, sin, log, exp, maximum, abs


def model(x):
    t = x[0]
    # lower branch: shifted sine plus linear trend
    m1 = -0.1128 + 0.5972 * sin(6.6687 * t - 3.0198) - 0.4451 * t
    # upper branch splits off with a smooth kink (softplus) near t = 0.634
    z = (t - 0.6341) / 0.09551
    m2 = m1 + 9.2879 * 0.09551 * (maximum(z, 0.0) + log(1.0 + exp(-abs(z))))
    s1 = exp(-1.2374 - 0.7786 * t)
    s2 = exp(-0.7883 - 1.3999 * t)
    w = 0.43891
    return Mixture([1.0 - w, w], [Normal(m1, s1), Normal(m2, s2)])
