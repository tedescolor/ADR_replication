from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    # lower (wavy, noisy) branch: cubic mean, log-quadratic sd
    m1 = -0.0269 + t * (-6.7042 + t * (19.5142 - 13.8361 * t))
    s1 = exp(-1.0382 + t * (0.4738 - 1.9224 * t))
    # upper (tight, linear) branch, switched on near x = 0.43
    m2 = 1.6219 + 5.0699 * (t - 0.8)
    s2 = exp(-1.792)
    w = sigmoid(0.2722) * sigmoid(50.0 * (t - 0.4257))
    return Mixture([1 - w, w], [Normal(m1, s1), Normal(m2, s2)])
