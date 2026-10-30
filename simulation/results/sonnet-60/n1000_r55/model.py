from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    u = x[0] - 0.5
    a = sigmoid((((0.4406)*u+-3.4784)*u+3.6828)*u+-0.9439)
    return Mixture(
        [a, 1 - a],
        [
            Normal(((((-5.1354)*u+-6.346)*u+-3.141)*u+1.9719)*u+-0.1036, exp(((-3.7026)*u+0.7472)*u+-1.77)),
            Normal(((((-1.4576)*u+-10.4478)*u+5.4088)*u+5.0512)*u+-0.0242, exp(((-0.9542)*u+-1.432)*u+-1.4302)),
        ],
    )
