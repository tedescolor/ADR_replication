# KCDE baseline on one fold of data/dataset_dq.csv (R package np, Hayfield and Racine 2008).
#
#   Rscript scripts/baselines/kcde_fit.R FOLD [data/dataset_dq.csv] [results/baselines]
#
# Conditional kernel density estimate of (y1, y2) given x1..x6 (npcdens), with second-order
# Gaussian kernels for the continuous variables and the Aitchison-Aitken kernel for the binary x4,
# and all bandwidths selected by likelihood cross-validation on the training rows (cv.ml, the
# default of npcdensbw). Folds: rep(1:5, each = ceiling(n/5)), as in the UCI study.
# Writes <out>/kcde_fold<k>_per_row.csv (fold, row, logpdf) and <out>/kcde_fold<k>.csv.
suppressPackageStartupMessages(library(np))
options(np.messages = FALSE)
args <- commandArgs(trailingOnly = TRUE)
k <- as.integer(args[1])
data_path <- if (length(args) >= 2) args[2] else "data/dataset_dq.csv"
out_dir <- if (length(args) >= 3) args[3] else "results/baselines"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

d <- read.csv(data_path)
n <- nrow(d)
fold <- rep(1:5, each = ceiling(n / 5))[1:n]
X <- data.frame(x1 = d$x1, x2 = d$x2, x3 = d$x3, x4 = factor(d$x4, levels = c(0, 1)),
                x5 = d$x5, x6 = d$x6)
Y <- data.frame(y1 = d$y1, y2 = d$y2)
tr <- fold != k
te <- fold == k

set.seed(20261023 + k)
t0 <- Sys.time()
bw <- npcdensbw(xdat = X[tr, ], ydat = Y[tr, ], bwmethod = "cv.ml")
fit <- npcdens(bws = bw, txdat = X[tr, ], tydat = Y[tr, ], exdat = X[te, ], eydat = Y[te, ])
dens <- fitted(fit)
minutes <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
lp <- log(dens)

write.csv(data.frame(fold = k, row = which(te) - 1, logpdf = lp),
          file.path(out_dir, sprintf("kcde_fold%d_per_row.csv", k)), row.names = FALSE)
write.csv(data.frame(fold = k, nll = -mean(lp), minutes = round(minutes, 2),
                     ybw = paste(signif(bw$ybw, 4), collapse = ";"),
                     xbw = paste(signif(bw$xbw, 4), collapse = ";")),
          file.path(out_dir, sprintf("kcde_fold%d.csv", k)), row.names = FALSE)
cat(sprintf("fold %d: test NLL %.4f (%.1f min)\n", k, -mean(lp), minutes))
