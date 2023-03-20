library(RcausalEGM)
if (!(reticulate::py_module_available('CausalEGM'))){
    skip("CausalEGM not available for testing")
}
#Generate a simple simulation data.
n <- 1000
p <- 10
v <- matrix(rnorm(n * p), n, p)
x <- rbinom(n, 1, 0.4 + 0.2 * (v[, 1] > 0))
y <- pmax(v[, 1], 0) * x + v[, 2] + pmin(v[, 3], 0) + rnorm(n)
model <- causalegm(x=x, y=y, v=v, n_iter=1000)
paste("The average treatment effect (ATE):", round(model$ATE, 2))

