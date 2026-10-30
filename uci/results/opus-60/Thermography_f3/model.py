from adr.dist import *


def model(x):
    """Latent left-skewed linear model for y; x24 ~ max(y, x1..x23), so y sits just below
    x24 with a probability that grows with the gap g = x24 - max(x1..x23)."""
    loc = 2.2025 + dot([-0.0052698, -0.034879, -0.11849, 0.060371, 0.10033, 0.044303], x[0:6]) + dot([0.019936, 0.090235, 0.020204, 0.096496, 0.23777, 0.098852], x[6:12]) + dot([-0.21207, -0.035196, -0.11557, -0.026786, -0.036276, -0.093931], x[12:18]) + dot([-0.052644, 0.079463, 0.11526, 0.012494, -0.039233, -0.022035], x[18:24]) + dot([0.75041], x[24:25])
    g = x[24] - maximum.reduce(x[1:24])
    eta = -2.896 + 3.7482 * tanh(39.525 * g) + -0.79057 * (x[24] - loc)
    w = sigmoid(eta)
    tight = SkewNormal(x[24] - 0.013382, 0.041357, -1.8801)
    broad = SkewNormal(loc, 0.61138 * exp(-0.19356 * tanh(20 * g)), -3.7647)
    return Mixture([w, 1 - w], [tight, broad])
