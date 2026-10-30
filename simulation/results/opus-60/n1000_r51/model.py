from adr.dist import Normal, Mixture, exp, softmax

def model(x):
    z = 2.0 * x[0] - 1.0
    m0 = ((((0.475403)*z+-1.64114)*z+1.03748)*z+2.72658)*z+-0.0344647
    s0 = exp((-0.664575)*z+-1.56999)
    m1 = ((((0.300356)*z+-1.45715)*z+-0.823958)*z+1.09963)*z+-0.0925507
    s1 = exp((-0.709881)*z+-1.50947)
    w = softmax(0.0, 0.0358886)
    return Mixture(w, [Normal(m0, s0), Normal(m1, s1)])
