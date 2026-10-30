from adr.dist import LogNormal, Normal, SkewNormal, Sequential, log, exp, where


def model(x):
    a = log(x[0])
    b = log(1 + x[1])
    c = log(1 + x[2])
    g = x[4] / 10
    h = log(x[5])
    q = (g - 1.4) ** 2
    a2 = (a - 1.6) ** 2
    h2 = (h - 1.6) ** 2
    ah = (a - 1.6) * (h - 1.6)
    first = LogNormal(0.7025 + 0.03591 * a + 0.0295 * c + 0.02428 * x[3] + 0.1786 * g + 0.1986 * h + 0.1581 * q + 0.05488 * ah - 0.02593 * h2, exp(-0.9605 - 0.09527 * a + 0.04341 * c - 0.1167 * x[3] - 0.2125 * g - 0.005402 * h + 0.02426 * h2))

    def cond(y0):
        ly = log(y0)

        return SkewNormal(0.1503 + 0.1282 * a + 0.05688 * b + 0.006148 * c + 0.05638 * g + 0.07644 * h + 0.2778 * q - 1.162 * ly + 0.3947 * y0 - 0.09869 * a2, exp(-0.4606 - 0.07139 * a - 0.09392 * g + 0.1096 * h + 0.2072 * ly), 1.269)

    return Sequential(first, cond)
