# Bandwidths of the kernel conditional density estimator (KCDE) shown in the
# univariate sample figure (replicate 43, n = 1000), computed with the R package np.
# Usage: Rscript scripts/kcde_bandwidths.R <training csv with columns x,y>
# The training set is the authors' replicate (scripts/common.py: training_set(1000, 43)).
# Output: results/kcde_n1000_r43.json, read by generate_images/generate_figures.py.
library(np)
options(np.messages = FALSE)
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[1])
# np defaults: likelihood cross-validation, second-order Gaussian kernels in x and y.
bw <- npcdensbw(ydat = d$y, xdat = d$x)
cat(sprintf('{"hx": %.10g, "hy": %.10g, "bwmethod": "%s", "kernel": "%s", "np_version": "%s"}\n',
            bw$xbw, bw$ybw, bw$pmethod, bw$cxkertype, as.character(packageVersion("np"))),
    file = "results/kcde_n1000_r43.json")
