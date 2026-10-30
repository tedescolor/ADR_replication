from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    x = x[0]
    sd = exp(-1.11878 + 0.23857 * x - 1.52723 * x**2)
    w = sigmoid(1.94906 - 2.28757 * x)
    a = Normal(0.14646 - 8.14272 * x + 20.61513 * x**2 - 10.07473 * x**3, sd)
    b = Normal(-0.4863 - 3.22177 * x + 13.25071 * x**2 - 10.56328 * x**3, sd)
    return Mixture([w, 1 - w], [a, b])
