"""Two-branch heteroscedastic mixture for y given x0 (u = 2*x0 - 1)."""
from adr.dist import Normal, StudentT, Mixture, exp, sigmoid


def model(x):
    u = 2.0 * x[0] - 1.0
    w = 0.514706 * sigmoid(19.0184 * (x[0] - 0.563439))
    lower = Normal((((((((-0.878559) * u + -0.937781) * u + 1.874183) * u + 2.097347) * u + -3.059174) * u + -1.637395) * u + 1.557928) * u + -0.052667, exp(((-0.160306) * u + -0.6828) * u + -1.417315))
    upper = Normal(((((-4.546094) * u + 7.748344) * u + -4.092489) * u + 3.278006) * u + 0.077197, exp((-1.083589) * u + -1.285076))
    return Mixture([1.0 - w, w], [lower, upper])
