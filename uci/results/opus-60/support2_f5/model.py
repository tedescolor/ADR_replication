from adr.dist import Normal, Mixture, softmax, log, exp, dot


def model(x):
    s = 0.02101
    f = [x[0], x[3], x[4], log(x[5]), x[19], x[22]]
    w = softmax(0, -1.583 + dot(f, [0.01244, 0.3766, 0.2941, 0.07641, 0.2537, 0.7204]), 0.6153 + dot(f, [-0.001646, 0.1109, -0.9494, -0.8394, 0.2129, -0.3949]),
                -0.3771 + dot(f, [0.004675, 0.07283, 0.6629, -0.3798, 0.1038, -0.6139]), -3.48 + dot(f, [0.01892, -0.3847, 0.2633, -0.4258, 0.2844, 2.176]))
    e = dot(f, [-0.0007766, 0.001995, 0.1018, 0.0537, 0.01851, 0.3467])
    e2 = e + e
    e4 = e2 + e2
    e6 = e4 + e2
    q6 = -1.105 + e6
    q7 = -1.508 + e6 + e
    v = softmax(e, -0.6017 + e2, -0.8483 + e2 + e, -1.292 + e4, -1.127 + e4 + e, log(exp(q6) + exp(q7)))
    hi = Mixture([exp(q6), exp(q7)], [Normal(6, s), Normal(7, s)])
    ints = Mixture([v[0], v[1], v[2], v[3], v[4] + v[5]],
                   [Normal(1, s), Normal(2, s), Normal(3, s), Normal(4, s),
                    Mixture([v[4], v[5]], [Normal(5, s), hi])])
    spec = Mixture([1, 0.3983, 0.5292, 0.117],
                   [Normal(0.4948, 0.01941), Normal(1.157, 0.03138), Normal(1.867, 0.0695), Normal(4.521, 0.02656)])
    m = 1.817 + dot(f, [0.005056, 0.08236, 0.1943, 0.1198, 0.1855, 1.2])
    return Mixture(w, [Normal(0, s), ints, spec, Normal(m, 0.4835), Normal(4.369, 1.725)])
