from adr.dist import Normal, Mixture, sigmoid, exp


def model(x):
    """Smooth curve up to x~0.6, then splits into an upper line and a lower parabola."""
    t = x[0]
    m0 = 0.0976083098 + t * (-14.5282695 + t * (104.725147 + t * (-370.925436
         + t * (651.950805 - 428.557367 * t))))
    s0 = exp(-0.744430951 - 1.37130921 * t)
    mu = -2.26964206 + 4.89973902 * t
    su = exp(-1.74456898)
    ml = -5.89178868 + t * (17.0673495 - 12.1330144 * t)
    sl = exp(0.161701946 - 2.55774088 * t)
    st = sigmoid(300.0 * (t - 0.600184259))
    q = sigmoid(-0.0961610682)
    return Mixture([1.0 - st + 1e-12, st * (1.0 - q), st * q],
                   [Normal(m0, s0), Normal(mu, su), Normal(ml, sl)])
