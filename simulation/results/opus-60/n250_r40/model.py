"""Two-branch mixture: a smooth quartic lower branch with log-linear sd, and a linear
upper branch (constant sd) whose weight rises logistically with x."""
from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    z = t - 0.5
    m1 = (((11.046226 * z - 15.076695) * z - 3.877681) * z + 2.613389) * z - 0.119527
    s1 = exp(-0.79132 - 1.404965 * t)
    m2 = 1.631008 + 4.7498 * (t - 0.8)
    s2 = exp(-1.851891)
    w = sigmoid(0.027299 + 4.571 * (t - 0.865464))
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
