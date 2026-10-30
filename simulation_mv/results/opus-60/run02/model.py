"""Conditional density of (y0, y1) given x: a mixture of
- a "line" through the origin at angle z = a.x (isotropic 2-scale Gaussian blobs at +-h(x) along it),
  with weight sigmoid(c0 + c1 z) and h = h0 + h1 tanh(b.x);
- a ring of radius g(z) = g0 + g1 tanh(g2 z + g3) with uniform angle, approximated by 25
  Gaussian blobs (5 x 5 nested mixtures), radial sd exp(r0 + r1 g), tangential sd k g.
"""
from adr.dist import MultivariateNormal, Mixture, sigmoid, tanh, cos, sin, exp, pi, dot


def blob(r, t, a, b):
    c = cos(t)
    s = sin(t)
    e = a * a - b * b
    m = b * b
    return MultivariateNormal([r * c, r * s], [[m + e * c * c, e * c * s], [e * c * s, m + e * s * s]])


def model(x):
    z = dot([-0.72126, 0.47872, -0.35982, 0.24079, -0.54194], x)
    w = sigmoid(0.0012 + 1.63352 * z)
    h = 1.98659 + 1.20422 * tanh(dot([-0.40002, 0.8693, -0.24089, 0.5633, 0.30708], x))
    sl = 0.1347
    so = 0.26331
    o = 0.65712
    g = 2.00996 + -1.45613 * tanh(1.1898 * z + 0.02988)
    sr = exp(-1.10207 + -0.12563 * g)
    kt = 0.13776 * g
    u = [1.0] * 5

    def b(t):
        return blob(g, t, sr, kt)

    q = 2 * pi / 25

    def arc(t):
        return Mixture(u, [b(t), b(t + 5 * q), b(t + 10 * q), b(t + 15 * q), b(t + 20 * q)])

    ring = Mixture(u, [arc(0), arc(q), arc(2 * q), arc(3 * q), arc(4 * q)])
    m = w * o
    n = w - m
    return Mixture([n, n, m, m, 2 - 2 * w],
                   [blob(h, z, sl, sl), blob(-h, z, sl, sl), blob(h, z, so, so), blob(-h, z, so, so), ring])
