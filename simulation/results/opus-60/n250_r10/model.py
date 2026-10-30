from adr.dist import StudentT, Normal, Mixture, sin, exp, sigmoid


def model(x):
    t = x[0]
    m = -0.1858 - 0.0555 * t + 0.598 * sin(7.5575 * t + 2.9874)
    s = exp(-1.0877 - 0.8092 * t)
    w = sigmoid(-0.0592) * sigmoid((t - 0.69) / 0.002)
    return Mixture([1.0 - w, w], [StudentT(14.83, m, s), Normal(-2.0801 + 4.6682 * t, 0.0965)])
