from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = -0.1566 + t * (-7.0104 + t * (25.3381 + t * (-24.9596 + t * 5.7454)))
    m2 = 0.3912 + t * (-13.1339 + t * (39.4453 + t * (-34.6583 + t * 10.4926)))
    s1 = exp(-0.9902 - 1.1505 * t)
    s2 = exp(-0.927 - 1.2989 * t)
    p = sigmoid(0.0709 - 0.1114 * t)
    return Mixture([p, 1 - p], [Normal(m1, s1), Normal(m2, s2)])
