from adr.dist import Normal, StudentT, Mixture, arctan2, sin, cos, exp, where, sigmoid, pi


def saw(t):
    return arctan2(sin(2 * pi * t), cos(2 * pi * t)) / (2 * pi)


def model(x):
    """y is linear in x3 when x3 is valid; when x3 reads ~0 (sensor dropout), predict from x0, x2."""
    x0, x2, x3 = x[0], x[2], x[3]
    # valid-x3 branch
    good = Normal(1.18031644 + 0.000999576061 * x3, 0.00294)
    # fallback: stepped structure in x2 (period 0.05), sawtooth in x0 (period 0.5), regime x0 > 4.55
    st2 = where(x2 > 0.095, x2 - 0.05 * saw(20 * x2 + 0.6), 0.07)
    s0 = saw(2 * x0 + 0.4)
    hi = where(x0 > 4.55, 1.0, 0.0)
    mu = (1.027056 + 0.024969 * x0 - 0.028906 * s0 + 0.934607 * st2 + 0.459241 * x2 + 0.005189 * hi
          - 0.02578 * x0 * s0 + 0.447815 * x0 * st2 - 0.126597 * x0 * x2 - 0.000619 * x0 * hi
          - 0.236477 * s0 * st2 + 0.357219 * s0 * x2 + 0.017368 * s0 * hi - 3.061586 * st2 * x2
          - 0.183901 * st2 * hi + 0.19636 * x2 * hi)
    sd = exp(-4.772469 - 0.090037 * x0)
    fallback = StudentT(exp(0.850486), mu, sd)
    w = 0.995 * sigmoid(x3 - 15)
    return Mixture([w, 1 - w], [good, fallback])
