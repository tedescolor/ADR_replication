from adr.dist import Normal, Mixture, exp, maximum, sigmoid


def model(x):
    """Two branches: a shared curve g(x) and a branch that splits off above x = c."""
    t = 2.0 * x[0] - 1.0
    r = maximum(x[0] - 0.4216103248166857, 0.0)
    g = ((((((-1.2159095522119654) * t + 0.11370244956692789) * t + 2.2962043250705917) * t + -1.960535838386713) * t + -1.5638593719382359) * t + 1.3670503787145543) * t + -0.08318886508614234
    up = g + r * ((8.35705232995968) * r + 1.2378956986697152)
    s = exp((((-0.4937048665490562) * t + -0.2717216333604138) * t + -0.337181235967977) * t + -1.4722758991428067)
    w = sigmoid(-0.09939445448148643)
    return Mixture([w, 1.0 - w], [Normal(up, s), Normal(g, s)])
