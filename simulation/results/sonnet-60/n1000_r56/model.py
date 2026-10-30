from adr.dist import Normal, Mixture, exp, sigmoid


def model(x):
    u = x[0] - 0.5
    p = sigmoid(((-20.87)*u+11.4)*u+-1.338)
    return Mixture(
        [1 - p, p],
        [Normal((((-10.37)*u+5.044)*u+5.08)*u+-0.008846, exp(((-0.8511)*u+-1.353)*u+-1.41)),
         Normal((((-13.46)*u+-0.4233)*u+1.349)*u+0.01393, exp(((-2.236)*u+-0.03851)*u+-1.727))],
    )
