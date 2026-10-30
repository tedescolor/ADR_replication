from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    x = x[0]
    z = ((-7.48)*x+11.0096)*x+-4.0315
    m1 = ((((-1.9829)*x+-1.3801)*x+7.0453)*x+0.4708)*x+-1.6825
    m2 = ((((8.9549)*x+-32.8909)*x+32.2322)*x+-9.4225)*x+0.0978
    s1 = exp(((-4.0406)*x+3.9665)*x+-2.454)
    s2 = exp(((-0.7202)*x+-0.5034)*x+-1.0838)
    p = sigmoid(z)
    return Mixture([p, 1 - p], [Normal(m1, s1), Normal(m2, s2)])
