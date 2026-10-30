#!/usr/bin/env python3
"""Reproduce manuscript illustrations from archived ADR programs; no fitting.

Run from any directory. Inputs are read-only. Outputs go to paper/figures.
The univariate illustration uses n=1000, replicate 43: the shortest Opus 5.5 program,
compared with the authors' CPFN and a KCDE (R package np bandwidths) on the same training set.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import platform
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", "/tmp/adr-paper-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np
import pandas as pd
import scipy
from scipy.special import erf
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261005
# Colour-blind-safe categorical colours (validated: lightness band, chroma, CVD and normal-vision separation).
COLORS = {"Truth": "#242424", "Sonnet": "#eda100", "Opus": "#2a78d6", "CPFN": "#d55181",
          "KCDE": "#009e73"}
CONFIGS = {"Sonnet": "sonnet-60", "Opus": "opus-60"}
NAMES = {"Sonnet": "ADR (Sonnet 5.5)", "Opus": "ADR (Opus 5.5)"}
INPUTS = set()
CHECKS = {}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 10, "axes.titlesize": 11, "legend.fontsize": 9,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 200})


def load_module(name, path, package=False):
    INPUTS.add(path)
    kw = {"submodule_search_locations": [str(path.parent)]} if package else {}
    spec = importlib.util.spec_from_file_location(name, path, **kw)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# All three frozen libraries must be identical before sharing one import.
libs = [ROOT / s / "adr_snapshot" / "dist.py"
        for s in ("simulation", "simulation_mv", "uci")]
assert len({p.read_bytes() for p in libs}) == 1
INPUTS.update(libs)
load_module("adr", ROOT / "simulation" / "adr_snapshot" / "__init__.py", True)
load_module("adr.dist", libs[0])
mv = load_module("adr_paper_mv", ROOT / "simulation_mv" / "scripts" / "common.py")


def model(study, config, run):
    path = ROOT / study / "results" / config / run / "model.py"
    return load_module(f"plot_{study}_{config.replace('-', '_')}_{run}", path).model


class CPFN:
    """NumPy port of generative.PushForward (CPFN repository) for an authors' trained network."""
    gelu = staticmethod(lambda z: .5*z*(1+erf(z/np.sqrt(2))))  # torch's exact GELU

    def __init__(self, path):
        INPUTS.add(path)
        self.P = {k: v.astype(float) for k, v in np.load(path).items()}
        self.q = self.P["w12"].shape[1]
        self.r = self.P["w41"].shape[0] // self.q

    def net(self, z, k, last_gelu):  # dlroms Dense layers 50-50-50-(r*q), reshaped to (r, q)
        for l in range(1, 5):
            z = z @ self.P[f"w{l}{k}"].T + self.P[f"b{l}{k}"]
            z = self.gelu(z) if l < 4 or last_gelu else z
        return z.reshape(len(z), self.r, self.q)

    def push(self, x, m, rng):
        """phi(x, U) for m latent draws U, before the kernel noise."""
        P = self.P
        Y = self.net((np.atleast_2d(x)-P["ex_xmean"])/P["ex_xsd"], 1, False)[0]
        U = self.net(rng.standard_normal((m, self.q)), 2, True)
        return (Y*U).sum(-2)*P["ex_ysd"] + P["ex_ymean"]

    def sample(self, x, m, rng):
        y = self.push(x, m, rng)
        return y + self.P["h"]*rng.standard_normal(y.shape)

    def pdf(self, y, x, rng, m=20000):
        """Smoothed density E_U[N(y; phi(x, U), h^2)], univariate case, by Monte Carlo over U."""
        h = float(self.P["h"].ravel()[0])
        centres = self.push(x, m, rng).ravel()
        return norm.pdf((y[:, None]-centres[None])/h).mean(1)/h


def save(fig, stem, dpi=None):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", dpi=dpi if ext == "png" else None,
                    metadata={"Creator": "ADR manuscript figure generator"} if ext == "pdf" else None)
    plt.close(fig)
    # Opaque RGB PNG: no alpha channel, hence no transparency mask once embedded in the manuscript.
    from PIL import Image
    Image.open(OUT / f"{stem}.png").convert("RGB").save(OUT / f"{stem}.png")


def uni_sample(x, rng):
    m1 = 10 * x * (x - .5) * (1.5 - x)
    m2 = 10 * x * (x - .5) * (.8 - x)
    m = np.where((x < .5) | (rng.random(len(x)) < .5), m1, m2)
    return m + .3 * (1.3 - x) * rng.normal(size=len(x))


def uni_density(y, x):
    m1, m2 = 10*x*(x-.5)*(1.5-x), 10*x*(x-.5)*(.8-x)
    s = .3*(1.3-x)
    return norm.pdf(y, m1, s) if x < .5 else .5*(norm.pdf(y, m1, s)+norm.pdf(y, m2, s))


def univariate():
    models = {label: model("simulation", config, "n1000_r43")
              for label, config in CONFIGS.items()}
    # The authors' CPFN for the same training set (replicate 43, n = 1000).
    cpfn = CPFN(ROOT / "simulation" / "cpfn" / "trained" / "branched1000ntrain_43.npz")
    rng = np.random.default_rng(SEED)
    x = rng.uniform(0, 1, 1000)
    samples = {"Truth": uni_sample(x, rng)}
    for j, (label, fn) in enumerate(models.items()):
        r = np.random.default_rng(SEED + 10 + j)
        samples[label] = np.array([float(np.asarray(fn(np.array([v])).sample(r, 1)).ravel()[0])
                                   for v in x])
    r = np.random.default_rng(SEED + 12)
    samples["CPFN"] = np.array([cpfn.sample([[v]], 1, r)[0, 0] for v in x])
    # KCDE (R package np, likelihood cross-validated Gaussian kernels) on the same training set:
    # p(y|x) = sum_i K_hx(x-X_i) K_hy(y-Y_i) / sum_i K_hx(x-X_i), sampled by drawing i with
    # probability proportional to K_hx(x-X_i) and adding N(0, hy^2) noise to Y_i.
    sys.path.insert(0, str(ROOT / "simulation" / "scripts"))
    from common import training_set
    xt, yt = training_set(1000, 43)
    path = ROOT / "simulation" / "results" / "kcde_n1000_r43.json"
    INPUTS.add(path)
    bw = json.loads(path.read_text())
    r = np.random.default_rng(SEED + 13)
    w = norm.pdf((x[:, None] - xt[None]) / bw["hx"])
    w /= w.sum(1, keepdims=True)
    idx = np.array([r.choice(len(xt), p=wi) for wi in w])
    samples["KCDE"] = yt[idx] + bw["hy"] * r.standard_normal(len(x))
    assert all(np.isfinite(y).all() for y in samples.values())
    title = {"Truth": "Truth", "CPFN": "CPFN", "KCDE": "KCDE"}
    shown = {k: v for k, v in samples.items() if k != "Sonnet"}
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), sharex=True, sharey=True,
                             layout="constrained")
    for ax, (label, y) in zip(axes.flat, shown.items()):
        ax.scatter(x, y, s=4, color=COLORS[label], alpha=.44, linewidths=0)
        ax.set(title=title.get(label, NAMES.get(label)), xlim=(0, 1))
    for ax in axes[1]:
        ax.set_xlabel("$x$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Response $y$")
    save(fig, "univariate_samples")

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.9), layout="constrained")
    y = np.linspace(-2.4, 2.4, 1400)
    for ax, x0 in zip(axes, [.4, .8]):
        ax.plot(y, uni_density(y, x0), color=COLORS["Truth"], lw=2, label="Truth")
        density = np.exp(models["Opus"](np.array([x0])).logpdf(y))
        assert np.isfinite(density).all()
        ax.plot(y, density, color=COLORS["Opus"], ls="-.", lw=1.7, label=NAMES["Opus"])
        density = cpfn.pdf(y, [[x0]], np.random.default_rng(SEED + 20 + int(10*x0)))
        assert np.isfinite(density).all()
        ax.plot(y, density, color=COLORS["CPFN"], ls=":", lw=1.9, label="CPFN")
        # KCDE density at x0: kernel-weighted average of N(Y_i, hy^2) densities
        w = norm.pdf((x0 - xt) / bw["hx"]); w /= w.sum()
        density = (w[None, :] * norm.pdf((y[:, None] - yt[None, :]) / bw["hy"])).sum(1) / bw["hy"]
        assert np.isfinite(density).all()
        ax.plot(y, density, color=COLORS["KCDE"], ls="--", lw=1.4, label="KCDE")
        ax.set(title=f"$x={x0:.1f}$", xlabel="Response $y$", ylabel="Conditional density",
               ylim=(0, None), xlim=(-2.4, 2.4))
    axes[1].legend(frameon=False)
    save(fig, "univariate_densities")

    path = ROOT / "simulation" / "results" / "per_rep.csv"
    INPUTS.add(path)
    data = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(4.6, 3.3), layout="constrained")
    for k, n in enumerate([250, 1000]):
        for j, (label, config) in enumerate(CONFIGS.items()):
            values = data.loc[(data.config == config) & (data.n == n), "AWD"].to_numpy()
            assert len(values) == 100 and np.isfinite(values).all()
            pos = k * 3 + j
            bp = ax.boxplot(values, positions=[pos], widths=.62, patch_artist=True,
                            showfliers=True, medianprops={"color": "black", "linewidth": 1},
                            flierprops={"marker": ".", "markersize": 3, "alpha": .6})
            bp["boxes"][0].set(facecolor=COLORS[label], alpha=.55)
    ax.set_xticks([.5, 3.5], ["$n=250$", "$n=1000$"])
    ax.set(ylabel="AWD", ylim=(0, None))
    ax.grid(axis="y", alpha=.16)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COLORS[k], alpha=.55, label=NAMES[k]) for k in CONFIGS],
                   frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.20), ncol=2)
    save(fig, "univariate_errors")
    CHECKS["univariate"] = {"replicate": 43, "training_n": 1000, "new_draws": 1000,
                             "cpfn": "authors' trained network for the same training set, NumPy port",
                             "kcde": "np::npcdensbw bandwidths (cv.ml, Gaussian kernels), same training set",
                             "all_draws_finite": True, "replicates_per_box": 100}


def bivariate():
    opus = model("simulation_mv", CONFIGS["Opus"], "run00")
    cpfn = CPFN(ROOT / "simulation_mv" / "data" / "cpfn_blobs.npz")
    path = ROOT / "simulation_mv" / "data" / "blobs_3ex_true.csv"
    INPUTS.add(path)
    X, truth = mv.three_x_set()
    grid = np.linspace(-4.5, 4.5, 350)  # even grid avoids the null-set singularity at zero
    yy1, yy2 = np.meshgrid(grid, grid)
    yy = np.column_stack([yy1.ravel(), yy2.ravel()])
    fig, axes = plt.subplots(4, 3, figsize=(8.6, 10.1), sharex=True, sharey=True,
                             layout="constrained")
    norm_color = LogNorm(vmin=.001, vmax=3)
    density_cmap = LinearSegmentedColormap.from_list(
        "white_blue_density", ["#ffffff", "#deebf7", "#9ecae1", "#4292c6", "#08519c", "#08306b"])
    density_cmap.set_under("white")  # opaque image on white: no transparency mask for viewers to mishandle
    labels = ["Ring", "Transition", "Two blobs"]
    rowlabels = ["True density", "True draws", NAMES["Opus"], "CPFN"]
    omitted = {}
    for j, (x, regime) in enumerate(zip(X, labels)):
        density = np.exp(mv.true_logpdf(yy, x)).reshape(yy1.shape)
        assert np.isfinite(density).all() and (density >= 0).all()
        axes[0, j].pcolormesh(grid, grid, density, norm=norm_color, cmap=density_cmap, shading="auto",
                              rasterized=True, edgecolors="none", linewidth=0, antialiased=False)
        axes[0, j].set_title(regime)
        # Preserve the exact Opus draws from the original two-agent illustration.
        draws = {"Truth": truth[j],
                 "Opus": np.asarray(opus(x).sample(np.random.default_rng(SEED+101+10*j), 1000)),
                 "CPFN": cpfn.sample(x, 1000, np.random.default_rng(SEED+102+10*j))}
        for i, (label, y) in enumerate(draws.items(), start=1):
            assert y.shape == (1000, 2) and np.isfinite(y).all()
            axes[i, j].scatter(y[:, 0], y[:, 1], s=3, color=COLORS[label], alpha=.42, linewidths=0)
            omitted[f"{regime}: {label}"] = int((np.abs(y) > 4.5).any(axis=1).sum())
        for i in range(4):
            axes[i, j].set(aspect="equal", xlim=(-4.5, 4.5), ylim=(-4.5, 4.5),
                           xticks=[-4, -2, 0, 2, 4], yticks=[-4, -2, 0, 2, 4])
            if j == 0:
                axes[i, j].set_ylabel(rowlabels[i] + "\n$y_2$")
            if i == 3:
                axes[i, j].set_xlabel("$y_1$")
    save(fig, "bivariate_simulation")
    CHECKS["bivariate"] = {"draws_per_panel": 1000, "axis_range": [-4.5, 4.5],
                            "displayed_agent": "Opus", "cpfn": "authors' trained network, NumPy port", "density_colormap": "white_blue_density",
                            "density_below_0.001": "white (no colour bar; darker = higher density)",
                            "draws_outside_axes": omitted, "all_draws_finite": True}


def bivariate_repeats():
    """Draws from the five Opus 5.5 programs fitted to the same bivariate training sample."""
    runs = [("run00", "First run"), ("run01", "Repetition 1"), ("run02", "Repetition 2"),
            ("run03", "Repetition 3"), ("run04", "Repetition 4")]
    progs = [(label, model("simulation_mv", CONFIGS["Opus"], run)) for run, label in runs]
    X, truth = mv.three_x_set()
    fig, axes = plt.subplots(6, 3, figsize=(7.4, 13.6), sharex=True, sharey=True,
                             layout="constrained")
    labels = ["Ring", "Transition", "Two blobs"]
    rowlabels = ["True draws"] + [label for label, _ in progs]
    for j, (x, regime) in enumerate(zip(X, labels)):
        axes[0, j].scatter(truth[j][:, 0], truth[j][:, 1], s=3, color=COLORS["Truth"], alpha=.42, linewidths=0)
        axes[0, j].set_title(regime)
        for i, (label, prog) in enumerate(progs, start=1):
            y = np.asarray(prog(x).sample(np.random.default_rng(SEED + 301 + 10 * i + j), 1000))
            assert y.shape == (1000, 2) and np.isfinite(y).all()
            axes[i, j].scatter(y[:, 0], y[:, 1], s=3, color=COLORS["Opus"], alpha=.42, linewidths=0)
        for i in range(6):
            axes[i, j].set(aspect="equal", xlim=(-4.5, 4.5), ylim=(-4.5, 4.5),
                           xticks=[-4, -2, 0, 2, 4], yticks=[-4, -2, 0, 2, 4])
            if j == 0:
                axes[i, j].set_ylabel(rowlabels[i] + "\n$y_2$")
            if i == 5:
                axes[i, j].set_xlabel("$y_1$")
    save(fig, "bivariate_repeats")
    CHECKS["bivariate_repeats"] = {"draws_per_panel": 1000, "runs": [r for r, _ in runs]}


PROGRAMS = {"univariate_opus.py": "simulation/results/opus-60/n1000_r43/model.py",
            "forestfires_opus.py": "uci/results/opus-60/forestfires_f2/model.py",
            "toxicity_opus.py": "uci/results/opus-60/toxicity_f1/model.py"}


def programs():
    """Copy the archived programs shown in the text, byte for byte, to paper/programs."""
    out = ROOT / "paper" / "programs"
    out.mkdir(exist_ok=True)
    for name, src in PROGRAMS.items():
        INPUTS.add(ROOT / src)
        (out / name).write_bytes((ROOT / src).read_bytes())
    CHECKS["programs"] = PROGRAMS


def main():
    univariate()
    bivariate()
    bivariate_repeats()
    programs()
    manifest = {"seed": SEED, "python": platform.python_version(),
                "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                             "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
                "checks": CHECKS,
                "inputs_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(INPUTS)},
                "selection": {"univariate": "Replicate 43, n=1000: the Opus 5.5 program with the fewest syntax-tree nodes among the 100 at n=1000 (Program 1); chosen by size, not score.",
                              "bivariate": "Archived Opus run only, chosen for illustration; both agents remain in the results tables. CPFN draws come from the authors' trained network."},
                "note": "These are new illustrative draws from archived fits, not a rerun of benchmark scoring."}
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(CHECKS, indent=2))


if __name__ == "__main__":
    main()
