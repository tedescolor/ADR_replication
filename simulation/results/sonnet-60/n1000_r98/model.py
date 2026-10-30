from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    u = x[0]
    m1 = ((((6.102)*u+-24.021)*u+30.823)*u+-10.617)*u+0.23876
    m2 = ((((15.03)*u+-46.527)*u+43.707)*u+-14.122)*u+0.97758
    s1 = exp((-1.1929)*u+-0.97151)
    s2 = exp((-1.4816)*u+-0.75497)
    w = sigmoid(((5.275)*u+-8.7179)*u+3.5014)
    return Mixture([w, 1 - w], [Normal(m1, s1), Normal(m2, s2)])
