from adr.dist import Normal, MultivariateNormal, Mixture, Sequential, sqrt, maximum, exp, dot, cos, sin, abs


def ring(mu, sig):
    s1 = sqrt(0.0079 * mu * mu + sig * sig)
    s0 = sqrt(0.2427 * mu * mu + sig * sig)
    first = Mixture([0.4267, 0.4267, 1.0], [Normal(0.9064 * mu, s1), Normal(-0.9064 * mu, s1), Normal(0.0, s0)])

    def cond(y0):
        m = sqrt(maximum(mu * mu - y0 * y0, 0.0))
        tau = sig * mu / sqrt(m * m + 0.8999 * sig * mu)
        return Mixture([0.5, 0.5], [Normal(m, tau), Normal(-m, tau)])
    return Sequential(first, cond)


def pair(mu, phi, g, w):
    c = cos(phi)
    s = sin(phi)
    cov = [[w + c * c * g, c * s * g], [c * s * g, w + s * s * g]]
    return Mixture([0.5, 0.5], [MultivariateNormal([mu * c, mu * s], cov), MultivariateNormal([-mu * c, -mu * s], cov)])


def model(x):
    u = dot(x, [0.9425, -0.6249, 0.4675, -0.318, 0.6975])
    v = dot(x, [-0.268, 0.5711, -0.1564, 0.3721, 0.2066])
    l1 = exp(0.2306 - 0.9914 * u)
    l2 = exp(2.942 * u + 0.1849 * v - 3.8251)
    m0 = abs(1.9247 + u - 0.0304 * v)
    m2 = 3.0042 + 0.1377 * u
    pr = pair(1.9879 + v, 3.141 - 0.7712 * u - 0.0087 * v, 0.0599, 0.0541)
    return Mixture([1.0, l1, l2], [ring(m0, 0.3195), pr, ring(m2, 0.2143)])
