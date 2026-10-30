"""Two-component Gaussian mixture: cubic means, shared quadratic log-sd, quadratic logit weight."""
from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    ma = ((-9.39422 * t + 18.9471) * t - 7.04323) * t - 0.0485191
    mb = ((-5.32651 * t + 3.12902) * t + 2.82758) * t - 1.55797
    s = exp((-1.68863 * t + 0.164381) * t - 1.09439)
    w = sigmoid((5.04092 * t - 9.19783) * t + 4.23293)
    return Mixture([w, 1.0 - w], [Normal(ma, s), Normal(mb, s)])
