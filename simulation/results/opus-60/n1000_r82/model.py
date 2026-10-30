from adr.dist import Normal, Mixture, exp


def model(x):
    """Two-branch Gaussian mixture with shared heteroscedastic noise."""
    t = 2.0 * x[0] - 1.0
    m1 = -0.0301354 + 2.59868 * t + 1.2541 * t**2 - 1.34725 * t**3
    m2 = -0.0654818 + 1.24136 * t - 0.844967 * t**2 - 1.9104 * t**3 + 0.566055 * t**4
    s = exp(-1.40633 - 0.660059 * t - 0.278482 * t**2)
    return Mixture([0.558736, 0.441264], [Normal(m1, s), Normal(m2, s)])
