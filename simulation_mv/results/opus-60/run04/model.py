from adr.dist import MultivariateNormal, Mixture, cos, sin, softmax, dot, pi


def blob(t, Q):
    c = cos(t)
    s = sin(t)
    R, a, d, h = Q
    return MultivariateNormal([R * c, R * s], [[a + d * s * s, -d * c * s], [-d * c * s, a + d * c * c]])


def spread(f, t, h, Q):
    return Mixture([h, h, h, h, h], [f(t, Q), f(t + h, Q), f(t + h + h, Q), f(t + h + h + h, Q),
                                     f(t + h + h + h + h, Q)])


def fan(t, Q):
    return spread(blob, t, Q[3], Q)


def ring(R, s):
    h = 2 * pi / 25
    return spread(fan, h - h, 5 * h, (R, s * s, (0.3 * h * R) ** 2, h))


def model(x):
    z = dot(x, [1.207, -1.252, 0.6137, -0.6821, 0.4778]) ** 2
    W = softmax(-0.06551 + dot(x, [1.214, -0.9348, 0.6525, -0.4998, 0.6982]) + 0.2163 * z, 0 * z,
                -1.974 + dot(x, [2.346, -1.04, 1.177, -0.3463, 2.157]) + 0.688 * z)
    R2 = 2.021 + dot(x, [-0.347, 0.7499, -0.2111, 0.4828, 0.2649])
    R3 = 3.172 + dot(x, [0.05958, -0.04251, 0.03601, -0.02589, 0.06125])
    return Mixture(W, [ring(1.759, 0.7105), ring(R2, 0.2098), ring(R3, 0.2464)])
