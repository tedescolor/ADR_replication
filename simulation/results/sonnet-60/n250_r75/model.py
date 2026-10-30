from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    lower = (((((15.20351)*t+-26.7032)*t+-3.39412)*t+21.9363)*t+-7.90661)*t+-0.03138
    upper = 4.63479 * t + -2.07175
    w = sigmoid(-1.57304 + 3.9307 * (t - 0.5))
    return Mixture(
        [1 - w, w],
        [Normal(lower, exp(-1.02193 + -1.04252 * t)), Normal(upper, exp(-1.86712))],
    )
