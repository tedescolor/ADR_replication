from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    x = x[0]
    z = (((7.9762)*x + -1.6886)*x + -11.99)*x + 6.316
    m1 = (((-9.7353)*x + 19.552)*x + -7.2926)*x + -0.011394
    m2 = (((-6.3724)*x + 2.6787)*x + 5.4586)*x + -2.7979
    s1 = exp((((0.059383)*x + -0.98157)*x + -0.49488)*x + -0.9444)
    s2 = exp((((-3.3063)*x + 0.71205)*x + 2.95)*x + -2.9164)
    return Mixture([sigmoid(z), sigmoid(-z)], [Normal(m1, s1), Normal(m2, s2)])
