from adr.dist import Mixture, Normal, Sequential, dot, softmax, sigmoid, log, sqrt, maximum


def ring(R):
    return Mixture([0.4086, 0.1518, 0.1518, 0.144, 0.144], [Normal(0, 0.5193 * R), Normal(0.9849 * R, 0.1229 * R), Normal(-0.9849 * R, 0.1229 * R), Normal(0.6686 * R, 0.1889 * R), Normal(-0.6686 * R, 0.1889 * R)])


def arc(R, s, y0, q):
    m = sqrt(maximum(R * R - y0 * y0, 0))
    d = s * R / sqrt(m * m + 1.526)
    return Mixture([q, 1 - q], [Normal(m, d), Normal(-m, d)])


def model(x):
    R0 = maximum(1.626 + dot(x, [0.4377, -0.2933, 0.2064, -0.1886, 0.421]), 0.05)
    R1 = maximum(1.947 + dot(x, [-0.2636, 0.5493, -0.1505, 0.3541, 0.1862]), 0.05)
    R2 = maximum(2.665 + dot(x, [0.2656, -0.1832, 0.143, -0.09427, 0.1979]), 0.05)
    z = softmax(0, 0.6842 + dot(x, [-0.5115, 0.5298, -0.2548, 0.3371, -0.1704]), -0.8358 + dot(x, [2.318, -1.271, 1.11, -0.4053, 1.963]))
    t = 0.185 + dot(x, [-0.05603, 0.03849, -0.07195, 0.04529, -0.03472])
    def cond(y0):
        w = softmax(log(z[0]) + ring(R0).logpdf(y0), log(z[1]) + ring(R1).logpdf(y0), log(z[2]) + ring(R2).logpdf(y0))
        q = sigmoid(t * y0)
        return Mixture(w, [arc(R0, 0.7807, y0, q), arc(R1, 0.4421, y0, q), arc(R2, 0.321, y0, q)])
    return Sequential(Mixture(z, [ring(R0), ring(R1), ring(R2)]), cond)
