from adr.dist import Normal, SkewNormal, StudentT, Mixture, sin, exp, sigmoid


def model(x):
    """Two-branch mixture: wavy lower branch everywhere, linear-ish upper branch switching on for larger x0."""
    x0 = x[0]
    m1 = 0.17123 + 0.9893 * sin(5.17946 * x0 + 3.90804) + -1.56159 * x0
    s1 = exp(-0.52038 + -1.39748 * x0)
    m2 = -3.67319 + 8.08871 * x0 + -1.84559 * x0 * x0
    s2 = exp(-0.34524 + -2.16341 * x0)
    w = sigmoid(0.49697) * sigmoid(-2121.64803 + 4609.31505 * x0)
    return Mixture([1.0 - w, w], [SkewNormal(m1, s1, 2.17185), Normal(m2, s2)])
