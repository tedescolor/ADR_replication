# Regenerates, with the authors' own R code (copied unchanged from Rcode/generate_multivariate_data.R, plotting removed),
# (a) the 3000 true samples at x_ring_dominant, x_transition, x_blobs_dom (set.seed(1234)): blobs_3ex_true.csv, and
# (b) the 30-point test set (set.seed(123)): blobs_test_regenerated.csv, identical to blobs_test.npy up to 1e-14.
# Needs R with package mvtnorm.  Usage: Rscript make_3ex.R
suppressPackageStartupMessages(library(mvtnorm))
logistic <- function(z) 1/(1+exp(-z))

rot2 <- function(theta) {
  cth <- cos(theta); sth <- sin(theta)
  matrix(c(cth, -sth, sth, cth), 2, 2)
}

# Truncated Normal pdf on [0, ∞) with mean=mu, sd=sig
dtruncnorm_pos <- function(r, mu, sig) {
  # pdf(r) for r >= 0, zero otherwise
  out <- numeric(length(r))
  C <- 1 - pnorm(0, mean = mu, sd = sig)
  idx <- r >= 0
  out[idx] <- dnorm(r[idx], mean = mu, sd = sig) / C
  out[!idx] <- 0
  out
}

# ----- Parametrization Y|X -----
# Map X (5D) to parameters that control shape/topology
map_params <- function(x) {
  stopifnot(length(x) == 5)
  # Two scores controlling different aspects
  beta  <- c(1.2, -0.8, 0.6, -0.4, 0.9)
  gamma <- c(-0.5, 1.1, -0.3, 0.7, 0.4)
  s1 <- sum(beta * x)    # drives mixture weight, ring radius, mean rotation
  s2 <- sum(gamma * x)   # drives ring thickness, blob spacing & covariance
  
  # Mixture weight: ring vs blobs
  w_ring <- logistic(s1)
  
  # Ring radius & thickness
  ring_r  <- 2.0 + 1.5 * tanh(0.7 * s1)                # ~ in [0.5, 3.5]
  ring_sd <- 0.18 + 0.15 * logistic(1.2 * s2)          # thickness ~[0.18, 0.33]
  
  # Two-blobs parameters
  mu_sep  <- 2.0 + 1.2 * tanh(0.8 * s2)                # spacing of the two modes
  theta_m <- 0.6 * s1                                  # rotate means with s1
  
  # Base eigenvalues for blob covariance, then rotate with s2
  ev1 <- 0.30^2
  ev2 <- 0.10^2 * (1 + 0.8 * logistic(s2))             # slightly fatter with s2
  theta_S <- 0.7 * s2
  Q <- rot2(theta_S)
  Sigma <- Q %*% diag(c(ev1, ev2)) %*% t(Q)
  
  # Means (±mu along a rotated axis)
  m_base <- c(mu_sep, 0)
  Mrot <- rot2(theta_m)
  m1 <- as.vector(Mrot %*%  m_base)
  m2 <- as.vector(Mrot %*% -m_base)
  
  list(w_ring = w_ring, ring_r = ring_r, ring_sd = ring_sd,
       m1 = m1, m2 = m2, Sigma = Sigma)
}

# ----- Component densities -----
# 1) Ring (annulus) density:
# If R ~ TruncNormal_pos(mean=ring_r, sd=ring_sd) and Angle ~ Uniform(0, 2π),
# then the joint density on y∈R^2\{0} is g(R)/(2πR), where R = ||y||.
dring <- function(y, ring_r, ring_sd) {
  r <- sqrt(rowSums(y^2))
  g <- dtruncnorm_pos(r, mu = ring_r, sig = ring_sd)
  # Avoid division by zero at the origin (measure zero anyway)
  dens <- ifelse(r > 0, g / (2*pi*r), 0)
  dens
}

# 2) Two-blobs Gaussian mixture (equal weights)
dtwo_blobs <- function(y, m1, m2, Sigma) {
  0.5 * mvtnorm::dmvnorm(y, mean = m1, sigma = Sigma) +
    0.5 * mvtnorm::dmvnorm(y, mean = m2, sigma = Sigma)
}

# ----- Full conditional density f_{Y|X}(y | x) -----
dY_given_X <- function(y, x) {
  p <- map_params(x)
  w  <- p$w_ring
  w * dring(y, p$ring_r, p$ring_sd) + (1 - w) * dtwo_blobs(y, p$m1, p$m2, p$Sigma)
}


rY_given_X <- function(n, x) {
  stopifnot(n >= 1)
  p <- map_params(x)
  w <- p$w_ring
  z <- rbinom(n, 1, w)  # 1 = ring, 0 = blobs
  
  Y <- matrix(NA_real_, nrow = n, ncol = 2)
  
  # ----- Ring draws -----
  ring_idx <- which(z == 1)
  n_ring <- length(ring_idx)
  if (n_ring > 0) {
    # Rejection sampler for truncated N(mu=p$ring_r, sd=p$ring_sd) on [0, ∞)
    R <- numeric(n_ring)
    filled <- 0L
    while (filled < n_ring) {
      # sample in batches for efficiency and to avoid pathological while-loops
      batch_size <- max(4 * (n_ring - filled), 50)
      cand <- rnorm(batch_size, mean = p$ring_r, sd = p$ring_sd)
      acc <- cand[cand >= 0]
      m <- min(length(acc), n_ring - filled)
      if (m > 0) {
        R[(filled + 1):(filled + m)] <- acc[seq_len(m)]
        filled <- filled + m
      }
    }
    A <- runif(n_ring, 0, 2 * pi)
    Y[ring_idx, ] <- cbind(R * cos(A), R * sin(A))
  }
  
  # ----- Blob draws -----
  blob_idx <- which(z == 0)
  n_blob <- length(blob_idx)
  if (n_blob > 0) {
    which_blob <- rbinom(n_blob, 1, 0.5)
    means <- rbind(p$m1, p$m2)
    mu_sel <- means[which_blob + 1, , drop = FALSE]
    Y[blob_idx, ] <- mvtnorm::rmvnorm(n_blob, mean = c(0, 0), sigma = p$Sigma) + mu_sel
  }
  
  Y
}

dist_X <- function(n) mvtnorm::rmvnorm(n, mean = rep(0, 5), sigma = diag(5))
x_ring_dominant <- c( 2.0,  1.0,  0.5, -0.3, -1.0)
x_transition    <- c( 0.0,  0.0,  0.0,  0.0,  0.0)
x_blobs_dom     <- c(-2.0, -1.0, -0.5,  0.3,  1.0)
# 3 examples (seed 1234), as in the authors' script
set.seed(1234)
samples <- matrix(0, nrow = 3000, ncol = 7)
configs <- list(x_ring_dominant, x_transition, x_blobs_dom)
row_idx <- seq(1, 3000, by = 1000)
for (i in seq_along(configs)) {
  x <- configs[[i]]; ys <- rY_given_X(1000, x); r <- row_idx[i]:(row_idx[i] + 999)
  samples[r, 1:2] <- ys; samples[r, 3:7] <- rep(x, each = 1000)
}
write.csv(samples, file = "blobs_3ex_true.csv", row.names = FALSE)
# 30-point test set (seed 123), as in the authors' script
set.seed(123); nx <- 30; X <- dist_X(nx); ny <- 1e3
df <- matrix(0, nrow = nx*ny, ncol = 7)
for(i in 1:nx){ idx <- ((i-1)*ny+1):(i*ny); df[idx,1:2] <- rY_given_X(ny, X[i, ]); df[idx,3:7] <- X[i,,drop=FALSE][rep(1,ny),] }
write.csv(df, "blobs_test_regenerated.csv", row.names = FALSE)
