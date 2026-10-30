from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    m1 = ((((-0.4761)*t+-7.9676)*t+17.543)*t+-6.5234)*t+-0.0837
    m2 = ((-10.522)*t+14.186)*t+-4.6226
    p = sigmoid(((-17.288)*t+27.903)*t+-10.967)
    s1 = exp((-1.3601)*t+-0.81985)
    s2 = exp((-2.2861)*t+-0.13141)
    return Mixture([1 - p, p], [Normal(m1, s1), Normal(m2, s2)])
