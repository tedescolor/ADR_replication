from adr.dist import Mixture, Normal, exp, sigmoid

def model(x):
    t = x[0]
    m1 = ((((-0.07525)*t + -9.345)*t + 19.42)*t + -7.516)*t + 0.06633
    s1 = (-1.211)*t + -0.9006
    m2 = ((((-18.04)*t + 43.03)*t + -44.74)*t + 23.65)*t + -4.927
    s2 = (-2.394)*t + -0.03078
    a = (((-69.11)*t + 144.2)*t + -98.26)*t + 22.06
    return Mixture([sigmoid(a), 1 - sigmoid(a)], [Normal(m1, exp(s1)), Normal(m2, exp(s2))])
