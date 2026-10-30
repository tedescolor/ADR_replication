from adr.dist import StudentT, exp


def model(x):
    u = x[0] - 0.5
    m = -0.076342 + u * (3.8598 + u * (0.46933 + u * (-10.661 + u * 8.3613)))
    s = exp(-1.2941 + u * (1.6533 + u * 5.3604))
    return StudentT(3000.2, m, s)
