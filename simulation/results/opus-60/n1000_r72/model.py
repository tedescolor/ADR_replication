from adr.dist import Normal, Mixture, exp, log


def model(x):
    t = 2 * x[0] - 1
    mid = -0.058 + t * (1.97 + t * (0.046 + t * (-1.608 + t * 0.401)))
    k = exp(2.475)
    half = 4.465 * log(1 + exp(k * (x[0] - 0.616))) / k
    s = 0.237 - 0.156 * t
    return Mixture([0.518, 0.482], [Normal(mid + half, s), Normal(mid - half, s)])
