from adr.dist import Mixture, Normal, exp, sigmoid


def model(x):
    u = x[0]
    p = sigmoid(((-15.7981)*u+25.32969)*u+-10.00355)
    return Mixture(
        [1 - p, p],
        [
            Normal(((((1.93526)*u+-13.81334)*u+22.30706)*u+-7.91503)*u+-0.00336, exp((-1.28915)*u+-0.86705)),
            Normal(((-10.8402)*u+14.69521)*u+-4.82433, exp((-2.11089)*u+-0.19754)),
        ],
    )
