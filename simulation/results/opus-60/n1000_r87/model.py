from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    """Two-branch Gaussian mixture: degree-6 polynomial means in t = 2*x0 - 1,
    shared log-quadratic noise scale, constant mixing weight."""
    t = 2.0 * x[0] - 1.0
    w = sigmoid(0.08923397918001162)
    s = exp(((-0.410818068762059) * t + -0.5067209954040388) * t + -1.5020373272256835)
    m1 = ((((((-0.3029010547647392) * t + -0.08774511619816619) * t + 0.9211978261941691) * t + -1.3661639479896288) * t + 0.6171928264296463) * t + 2.7507012799801784) * t + -0.014319470898004005
    m2 = ((((((-1.2528656426500258) * t + 0.017461922738409498) * t + 2.099859091616355) * t + -1.8502426403546721) * t + -1.2039636210425924) * t + 1.1569363955917884) * t + -0.07514208935305985
    return Mixture([w, 1.0 - w], [Normal(m1, s), Normal(m2, s)])
