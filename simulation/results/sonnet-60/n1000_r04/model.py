from adr.dist import Mixture, Normal
import math


def model(x):
    t = x[0]
    z = 1.745 - 3.2506 * t + 1.2289 * t * t
    w = 1 / (1 + math.exp(-z))
    m1 = ((-12.0529 * t + 23.4737) * t - 9.0767) * t + 0.1148
    m2 = ((-9.9361 * t + 12.7824) * t - 3.5708) * t - 0.2436
    s1 = math.exp(-1.315 * t - 0.9123)
    s2 = math.exp(-1.2768 * t - 0.9085)
    return Mixture([w, 1 - w], [Normal(m1, s1), Normal(m2, s2)])
