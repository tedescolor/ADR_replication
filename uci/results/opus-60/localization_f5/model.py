from adr.dist import StudentT, exp, log


def model(x):
    m = exp(1.03129 - 0.0094303 * x[0] - 0.0660263 * x[1] - 0.00634048 * x[2]
            - 0.00228408 * x[3] + 0.644549 * log(x[4]))
    return StudentT(3.95257, m, 0.238209 * m)
