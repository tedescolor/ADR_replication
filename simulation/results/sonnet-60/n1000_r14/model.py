from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    t = x[0]
    wb = sigmoid(-6.651 + 18.45*t + -12.35*t**2)
    return Mixture(
        [1 - wb, wb],
        [
            Normal(-0.03212 + -7.362*t + 21.33*t**2 + -8.955*t**3 + -14.27*t**4 + 8.3*t**5, exp(-0.9613 + -0.5686*t + -0.692*t**2)),
            Normal(-3.146 + 7.071*t + -1.356*t**2, exp(-0.7436 + -1.513*t)),
        ],
    )
