"""Two-branch normal mixture: shared cubic base curve that splits at x0 = t."""
from adr.dist import Normal, Mixture, exp, maximum, sigmoid


def model(x):
    z = 2 * x[0] - 1
    f = 0.0418985 + z * (2.85943 + z * (1.93472 + z * (-0.847165)))
    h = maximum(x[0] - 0.452715, 0)
    m1 = f - 4.92537 * h * h
    m2 = f + h * (-2.06278 + -12.8453 * h)
    s = exp(-1.53742 + -0.679548 * z)
    w = sigmoid(0.165202)
    return Mixture([w, 1 - w], [Normal(m1, s), Normal(m2, s)])
