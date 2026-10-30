from adr.dist import Normal, Mixture, sigmoid, abs


def model(x):
    t = x[0] - 0.5
    ma = ((((((-97.5629 * t + 26.5053) * t + 44.503) * t - 23.5692) * t - 7.0665) * t + 3.2988) * t - 0.0646)
    mb = 0.2198 + 4.7612 * t
    s = abs(0.0904 + 0.2712 * (0.5 - t)) + 0.0001
    w = 0.5 * sigmoid(-1.1678 + 19.5369 * t)
    return Mixture([1.0 - w, w], [Normal(ma, s), Normal(mb, s)])
