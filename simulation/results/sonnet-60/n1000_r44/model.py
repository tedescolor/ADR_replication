from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0] - 0.5
    mu = ((-12.3455 * t + 1.4968) * t + 3.9261) * t - 0.11634
    s = exp((4.39827 * t + 1.2966) * t - 1.17923)
    q = sigmoid(-0.28733)
    k = 0.58655
    return Mixture([q, 1 - q], [Normal(mu - 0.87893 * s, s * k), Normal(mu + 0.87893 * s, s * k)])
