from adr.dist import Normal, Mixture, exp, sigmoid, maximum


def model(x):
    t = x[0]
    m1 = ((-9.9843 * t + 20.3119) * t - 7.8668) * t + 0.0597
    m2 = (-10.7818 * t + 14.997) * t - 5.104
    s = maximum(0.3935 - 0.3094 * t, 0.05)
    q = sigmoid(-0.1577) * sigmoid(59.7 * (t - 0.5782))
    return Mixture([1 - q, q], [Normal(m1, s), Normal(m2, s)])
