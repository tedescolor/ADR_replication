from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    x0 = x[0]
    w = sigmoid((((53.9521)*x0+-119.8276)*x0+85.4383)*x0+-19.4478)
    a = Normal((((((-0.2345)*x0+2.9165)*x0+-14.5711)*x0+21.8618)*x0+-7.3561)*x0+-0.0808, exp(-1.5081 * x0 + -0.824))
    b = Normal(((-10.1909)*x0+13.6238)*x0+-4.381, exp(-1.4438 * x0 + -0.8016))
    return Mixture([1 - w, w], [a, b])
