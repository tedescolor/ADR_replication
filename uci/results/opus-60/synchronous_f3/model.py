"""Two-regime model.

x3 is either a precise linear readout of y (most rows) or a dropout near 0 (|x3| < ~5).
Readout regime: y ~ Normal(a + b*x3 + c*x0, s).
Dropout regime: y ~ Normal(cubic(x0, x2), exp(quadratic(x0, x2))).
A logistic switch in x3 blends them; a small floor weight on the fallback guards outliers.
"""
from adr.dist import Normal, Mixture, sigmoid, exp


def model(x):
    x0, x2, x3 = x[0], x[2], x[3]
    u = x0 - 4.5
    v = (x2 - 0.17) * 10.0
    mu_f = (1.51999284 + 0.0727354486 * u + 0.191380174 * v + 0.0288463802 * u * v
            + 0.00412394459 * v * v - 0.000829846455 * u * u - 0.024205897 * v ** 3
            + 0.00439711939 * u * v * v + 0.000752588272 * u * u * v)
    s_f = exp(-2.94409708 + 0.166405095 * u - 0.0714624642 * v - 0.135365148 * v * v)
    mu_r = 1.17917564 + 0.00099839255 * x3 + 0.00035665794 * x0
    w = sigmoid((x3 - 20.0) * 0.5) * 0.997
    return Mixture([w, 1.0 - w], [Normal(mu_r, 0.002845), Normal(mu_f, s_f)])
