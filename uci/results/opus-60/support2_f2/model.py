from adr.dist import Normal, Mixture, log, exp, where


def model(x):
    s = 0.0202
    h = 0.5
    one = h + h
    z = h - h
    r5 = log(x[5])
    r6 = log(x[6] + 15)
    a0, a1, a2 = Normal(z, s), Normal(one, s), Normal(one + one, s)
    a3, a4, a5 = Normal(one + 2, s), Normal(one + 3, s), Normal(one + 4, s)
    a6, a7 = Normal(one + 5, s), Normal(one + 6, s)
    p = exp(-0.229 - 0.1953 * x[19] - 0.5196 * x[2])
    q = exp(-1.939 - 1.074 * r5 + 0.294 * r6 - 1.364 * x[4] - 3.708 * x[11])
    c = exp(2.904 - 2.299 * r5 - 0.2756 * x[19])
    k1 = Normal(1.635 + 0.2565 * r5 + 0.1583 * x[19], 0.3393)
    k2 = Normal(2.568 + 0.2647 * x[19], 0.531)
    g = Mixture([0.2579, 0.1502], [Mixture([0.1276, 0.0214, 0.0106, 0.0614, 0.0369], [a1, Normal(1.165, s), Normal(1.84, s), a2, a3]),
                                    Mixture([0.0284, 0.0399, 0.0490, 0.0328], [a4, a5, a6, a7])])
    main = Mixture([p, q, c, 0.155, 0.4081], [a0, Normal(h, s), k1, k2, g])
    u = Mixture([0.335, 0.664], [Mixture([0.061, 0.092, 0.035, 0.052, 0.043], [a0, a1, a2, a3, a4]),
                                 Mixture([0.053, 0.134, 0.177, 0.353], [a5, a6, a7, Normal(5.23, 0.92)])])
    t = where(x[22] > h, one, z)
    return Mixture([one - t, t], [main, u])
