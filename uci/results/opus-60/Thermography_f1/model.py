from adr.dist import Normal, Mixture, exp, maximum, sigmoid


def model(x):
    """y = x24 - D. D: narrow spike near 0.034 (likely when x24 clearly exceeds the max of
    x1..x23) plus a 3-normal tail. h is a soft step in the gap g = x24 - max(x1..x23)."""
    m = maximum(maximum(maximum(x[1], x[2]), maximum(x[5], x[8])), maximum(maximum(x[12], x[15]), maximum(x[16], x[22])))
    m = maximum(m, maximum(maximum(maximum(x[3], x[4]), maximum(x[6], x[7])), maximum(maximum(x[9], x[10]), maximum(x[11], x[13]))))
    m = maximum(m, maximum(maximum(maximum(x[14], x[17]), maximum(x[18], x[19])), maximum(maximum(x[20], x[21]), x[23])))
    h = sigmoid((x[24] - m - 0.015) / 0.00195)
    t = x[24] + 0.051 * h
    return Mixture(
        [exp(3.709 * h), exp(2.865 - 0.835 * h), exp(2.128 + 0.517 * h), exp(1.827 - 0.391 * h)],
        [Normal(x[24] - 0.034, exp(-3.806)),
         Normal(t - 0.653, exp(-1.388)),
         Normal(t - 0.241, exp(-2.177)),
         Normal(t - 1.231, exp(-0.671))])
