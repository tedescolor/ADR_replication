"""Two-branch Gaussian mixture: upper branch everywhere, lower branch switching on near x~0.615."""
from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = 2 * x[0] - 1
    m1 = -0.018156 + t * (2.3768 + t * (1.5696 + t * (-0.71505 + t * (-0.95498 + t * (-0.4008 + t * (0.64967))))))
    m2 = -0.3495 + t * (2.6901 + t * (-3.8649 + t * (0.5662)))
    s1 = exp(-1.4558 + t * (-0.68103 + t * (-0.23313)))
    s2 = s1 * 0.98492
    pl = sigmoid(0.22778) * sigmoid((x[0] - 0.61477) * 65.012)
    return Mixture([1 - pl, pl], [Normal(m1, s1), Normal(m2, s2)])
