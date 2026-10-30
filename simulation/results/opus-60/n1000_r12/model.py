from adr.dist import Normal, Mixture, exp, sin, sigmoid


def model(x):
    """Two-branch Gaussian mixture; branch B switches on smoothly with x."""
    t = 2.0 * x[0] - 1.0
    mA = (2.429223498531978 * t + -4.0114565321526) * t + -3.080444257115247 + 4.620577607267872 * sin(1.5172951584581937 * t + 0.7032375226424448)
    mB = (-0.72029145427114 * t + 3.346591667164929) * t + -0.07309899515784271
    sd = exp(-0.6718371727827398 * t + -1.538441887026488)
    w = 0.5 * sigmoid(41.609321038612194 * (x[0] - 0.4011405881240528))
    return Mixture([1.0 - w, w], [Normal(mA, sd), Normal(mB, sd)])
