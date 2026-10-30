from adr.dist import Normal, Mixture, exp, sigmoid

def model(x):
    u = x[0]
    m1 = (((-9.8638)*u + 20.097)*u + -7.7723)*u + 0.063522
    s1 = (-1.4001)*u + -0.8075
    m2 = (((-10.269)*u + 13.07)*u + -3.5286)*u + -0.28872
    s2 = (-1.6126)*u + -0.60513
    z = (-2.5959)*u + 2.3317
    return Mixture([sigmoid(z), 1 - sigmoid(z)], [Normal(m1, exp(s1)), Normal(m2, exp(s2))])
