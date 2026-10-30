from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    u = x[0] - 0.5
    m1 = -0.0926 + u * (2.5857 + u * (-3.7061 + u * (-14.8607 + u * 9.0755)))
    m2 = 0.1637 + u * (5.2729 - 0.9976 * u)
    w = sigmoid(-1.2533 + 3.6403 * u)
    return Mixture([1 - w, w], [Normal(m1, exp(-1.4799 - 1.1950 * u)),
                                Normal(m2, exp(-1.5138 - 1.4571 * u))])
