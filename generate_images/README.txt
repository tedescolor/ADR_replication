REPRODUCIBLE ADR MANUSCRIPT FIGURES AND TABLES
============================================

From the ADR repository root, with Python 3.12 and the packages in
requirements.txt installed:
    python generate_images/generate_figures.py
    python generate_images/generate_tables.py

The outputs are written to paper/figures/, paper/tables/ and paper/programs/,
which are created if needed. The LaTeX source of the manuscript is not part of
this repository.

Optional environment setup:
    python3 -m venv .venv-paper
    .venv-paper/bin/pip install -r generate_images/requirements.txt
    .venv-paper/bin/python generate_images/generate_figures.py
    .venv-paper/bin/python generate_images/generate_tables.py

generate_figures.py
  Writes PDF and PNG versions of:
  - univariate_samples: 2x2 panels with the truth, ADR (Opus 5.5), CPFN and KCDE; n=1000,
    replicate 43 (the Opus 5.5 program shown in the manuscript, the shortest by AST nodes).
    CPFN is the authors' network trained on the same data
    (simulation/cpfn/trained/branched1000ntrain_43.npz), sampled through a NumPy port.
    KCDE is the kernel conditional density estimator of the R package np on the same
    training set, with Gaussian kernels and bandwidths selected by likelihood
    cross-validation; the bandwidths are read from simulation/results/kcde_n1000_r43.json
    (written by simulation/scripts/kcde_bandwidths.R) and the draws are sampled exactly
    from the kernel estimate (an observation drawn with kernel weights in x, plus normal
    noise with the y bandwidth);
  - univariate_densities: ADR (Opus 5.5) density (direct evaluation) at x=0.4 and x=0.8, the
    CPFN smoothed density (Monte Carlo over 20,000 latent draws) and the KCDE density
    (closed form, same bandwidths);
  - univariate_errors: all 100 replicate errors per agent and sample size;
  - bivariate_simulation: true density on white and true/Opus/CPFN samples at
    three regimes. CPFN draws come from the authors' trained network
    (simulation_mv/data/cpfn_blobs.npz) through a NumPy port of its sampler. Sonnet is omitted from this illustration; its results remain in
    the manuscript tables. The original Opus sampling seeds are preserved.
  Recomputing the KCDE bandwidths requires R with the package np:
      Rscript simulation/scripts/kcde_bandwidths.R <training csv with columns x,y>
  run from simulation/, where the training csv is simulation/scripts/common.py:training_set(1000, 43).
  It also copies the three Opus programs shown in the text, unchanged, to
  paper/programs/ (univariate n1000_r43, the shortest by AST nodes; UCI forestfires_f2 and toxicity_f1).
  Output directory: paper/figures/.
  The base seed is 20261005. Illustrative draws are separate from archived
  benchmark-scoring draws. One bivariate Opus draw is outside the common
  display range; the figure caption and manifest disclose it.
  No model is fitted, no agent is called, and no experiment file is changed.

generate_tables.py
  Writes all manuscript table fragments into paper/tables/.
  Reads the released per-run CSVs and designated published baseline sources.
  Checks the number of replicates, agreement of univariate ADR summaries,
  and the identity of the single failed UCI fold. The UCI table reports
  ADR (Opus 5.5) only; mean ranks use all 12 datasets, from the means rounded to
  two decimals with average ranks for ties. Means used for ranks are also exported.
  The appendix table univariate_exact reads simulation/results/per_rep_exact.csv
  (simulation/scripts/score_exact.py: ADR errors with exact quantiles).

Manifests record input SHA256 hashes and generation settings. The checked-in
PDF figures and LaTeX tables can be used without regenerating them. The
figure environment is separate from the frozen environments used to obtain
the archived experimental scores; software versions are recorded explicitly.

The scripts need the original ADR repository's simulation/, simulation_mv/,
and uci/ directories at their existing locations. A portable manuscript-only
source archive can compile its bundled assets but cannot regenerate them
without these original inputs.
