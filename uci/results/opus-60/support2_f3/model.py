from adr.dist import Normal, Mixture, exp, log, abs, dot


def w(a, b, c, s, t):
    """mixture weights exp(a + b * s + c * t) for a group of components"""
    return exp(a) * exp(b) ** s * exp(c) ** t


def model(x):
    """Gated mixture: narrow normals (shared sd) at the discrete values of y plus 4 broad
    normals; component log-weights are a_k + b_k * s + c_k * t, s and t linear scores."""
    f = [log(x[5]), log(abs(x[6]) + 1), x[22], x[19], x[0], x[3]]
    s = dot([-0.5819, -0.3491, 0.07072, 0.3981, 0.01483, 0.5949], f) + 0.1511 * x[15]
    t = dot([-2.463, 0.2417, -14.53, -0.4682, -0.02747, -0.4694], f) - 0.5691 * x[15]
    sa = 0.02021
    w0 = w([1.289, -1.576, -0.8461, -0.432, -2.38], [0.2153, 0.3295, 0.3551, 0.0749, 0.374], [0.4067, -0.1089, 0.2151, 0.4031, -0.1195], s, t)
    w1 = w([-2.826, -3.326, -4.18, -2.946, -3.14], [0.5443, 0.4738, 0.7428, 0.5421, 0.475], [-0.146, -0.1646, -0.04469, -0.1613, -0.1925], s, t)
    w2 = w([-3.924, -0.7949, -1.882, -6.53], [0.5237, 0.9936, -0.09377, 0.9635], [-0.2469, -0.01963, -0.04087, -0.3885], s, t)
    wc = exp(5.826 + s + t)
    m0 = Mixture(w0, [Normal(0.5, sa), Normal(1, sa), Normal(1.16667, sa), Normal(1.83333, sa), Normal(2, sa)])
    m1 = Mixture(w1, [Normal(3, sa), Normal(4, sa), Normal(4.5, sa), Normal(5, sa), Normal(6, sa)])
    m2 = Mixture([w2[0], w2[1], w2[2], w2[3], wc], [Normal(7, sa), Normal(3.455, 0.5273), Normal(2.661, 0.3976), Normal(5.198, 0.7553), Normal(2.228, 0.3518)])
    return Mixture([1.0, w0.sum(), w1.sum(), w2.sum() + wc], [Normal(0, sa), m0, m1, m2])
