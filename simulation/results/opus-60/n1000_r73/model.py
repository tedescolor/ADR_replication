"""Two-branch Gaussian mixture: equal weights, quartic means in t = 2*x0 - 1,
shared noise sd linear in t."""
from adr.dist import Normal, Mixture, maximum


def model(x):
    t = 2.0 * x[0] - 1.0
    m1 = ((((0.34668) * t + -1.681521) * t + -0.731533) * t + 1.126669) * t + -0.060794
    m2 = ((((0.617619) * t + -1.565086) * t + 0.661017) * t + 2.821524) * t + 0.016717
    s = maximum(0.212609 + -0.109768 * t, 0.03)
    return Mixture([0.5, 0.5], [Normal(m1, s), Normal(m2, s)])
