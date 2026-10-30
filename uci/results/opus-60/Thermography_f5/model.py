from adr.dist import Mixture, SkewNormal, maximum, minimum, exp


def model(x):
    m = maximum(maximum(maximum(maximum(maximum(maximum(maximum(maximum(maximum(maximum(maximum(maximum(maximum(maximum(maximum(maximum(maximum(maximum(maximum(maximum(maximum(maximum(x[1], x[2]), x[3]), x[4]), x[5]), x[6]), x[7]), x[8]), x[9]), x[10]), x[11]), x[12]), x[13]), x[14]), x[15]), x[16]), x[17]), x[18]), x[19]), x[20]), x[21]), x[22]), x[23])
    g = minimum(maximum(x[24] - m, -0.224), 1.17)
    return Mixture([0.8612, 0.1388], [
        SkewNormal(x[24] + 0.0034201, exp(-0.46553 + -3.7542 * g), -56.848),
        SkewNormal(x[24] + -0.40172, exp(-0.14069 + -0.54296 * g), -70.897),
    ])
