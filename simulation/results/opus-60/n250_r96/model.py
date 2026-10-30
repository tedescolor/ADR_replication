from adr.dist import Normal, Mixture, exp, sin, sigmoid


def model(x):
    t = x[0]
    mu_a = 0.2358 - 1.0382 * t + 0.7567 * sin(6.182 * t + 3.4279)
    sd_a = exp(-0.8558 - 1.2233 * t)
    mu_b = -3.6822 + t * (8.6081 - 2.3748 * t)
    sd_b = exp(-2.053)
    w_b = 0.5 * sigmoid(75.8287 * (t - 0.5601))
    return Mixture([1.0 - w_b, w_b], [Normal(mu_a, sd_a), Normal(mu_b, sd_b)])
