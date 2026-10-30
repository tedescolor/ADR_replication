"""Mixture of a ring and a bar, driven by two linear indices u and v of x.
Ring (weight w): circle of radius R(u) with uniform angle plus N(0, s I) noise, s = s(v);
noise is a scale mixture (variance s with prob pi, l * s otherwise) in ring and bar;
the uniform angle is discretised into 25 Gaussians stretched along the tangent.
Bar (weight 1 - w): +-m(v) along direction u (radians), isotropic Gaussian noise."""
from adr.dist import MultivariateNormal, Mixture, cos, sin, exp, dot, sigmoid


def model(x):
    u = dot([-0.7209, 0.4785, -0.3600, 0.2399, -0.5408], x)
    v = 0.4202 + dot([-0.4828, 1.0502, -0.2934, 0.6762, 0.3741], x)
    R = 3.5077 * sigmoid(0.2746 - 1.9615 * u)
    w = sigmoid(-1.6322 * u)
    m = 3.3954 * sigmoid(v)
    s = exp(-3.8925 + 0.3531 * v)
    j = R * R * 0.02124
    g, h = m * cos(u), m * sin(u)

    def comp(t, s):
        q, z = cos(t), sin(t)
        o = -j * q * z
        return MultivariateNormal([R * q, R * z], [[s + j * z * z, o], [o, s + j * q * q]])

    def five(t, s):
        return Mixture([1] * 5, [comp(t, s), comp(t + 1.2566371, s), comp(t + 2.5132741, s),
                                 comp(t + 3.7699112, s), comp(t + 5.0265482, s)])

    def ring(s):
        return Mixture([1] * 5, [five(0, s), five(0.2513274, s), five(0.5026548, s),
                                 five(0.7539822, s), five(1.0053096, s)])

    def bar(s):
        return Mixture([1, 1], [MultivariateNormal([g, h], [[s, 0], [0, s]]),
                                MultivariateNormal([-g, -h], [[s, 0], [0, s]])])

    return Mixture([w * 0.2287, w * 0.7713, (1 - w) * 0.2287, (1 - w) * 0.7713],
                   [ring(s), ring(s * 3.5047), bar(0.01945), bar(0.06815)])
