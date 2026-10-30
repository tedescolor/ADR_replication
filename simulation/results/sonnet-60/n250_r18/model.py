from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    x0 = x[0]
    p = sigmoid(((0.9374)*x0+(-0.7354))*x0+(0.2773))
    return Mixture(
        [p, 1 - p],
        [
            Normal(((((-8.877)*x0+(12.03))*x0+(1.05))*x0+(-1.342))*x0+(-0.4048), exp((-0.8192)*x0+(-1.316))),
            Normal(((((18.55)*x0+(-57.01))*x0+(52.83))*x0+(-15.79))*x0+(0.4267), exp((-1.911)*x0+(-0.7423))),
        ],
    )
