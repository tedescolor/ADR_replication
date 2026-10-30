from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    x = x[0]
    m1 = ((((10.848)*x+-37.032)*x+35.488)*x+-10.4)*x+0.13146
    m2 = (5.1083)*x+-2.4711
    z = (((43.874)*x+-92.696)*x+63.12)*x+-13.667
    s1 = exp(((-1.3395)*x+0.051627)*x+-1.0877)
    s2 = exp(((-1.2903)*x+0.68286)*x+-1.5241)
    p = sigmoid(z)
    return Mixture([1 - p, p], [Normal(m1, s1), Normal(m2, s2)])
