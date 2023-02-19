#' @title Make predictions with causalEGM model.
#' @description When x is NULL, the conditional average treatment effect (CATE), namely tau(v), is estimated using a trained causalEGM model.
#' When x is provided, estimating the potential outcome y given treatment x and covariates v using a trained causalEGM model.
#'
#' @param object The trained causalEGM object.
#' @param v is the covariates, two-dimensional array with size n by p.
#' @param x is the optional treatment variable, one-dimensional array with size n. Defaults to NULL.
#' @return Vector of predictions.
#'
#' @examples
#' \donttest{
#' #Generate a simple simulation data.
#' n <- 1000
#' p <- 10
#' v <- matrix(rnorm(n * p), n, p)
#' x <- rbinom(n, 1, 0.4 + 0.2 * (v[, 1] > 0))
#' y <- pmax(v[, 1], 0) * x + v[, 2] + pmin(v[, 3], 0) + rnorm(n)
#' model <- causalegm(x=x, y=y, v=v, n_iter=3000)
#' n_test <- 100
#' v_test <- matrix(rnorm(n_test * p), n_test, p)
#' x_test <- rbinom(n_test, 1, 0.4 + 0.2 * (v_test[, 1] > 0))
#' pred <- predict(model, x_test, v_test)
#' }
#' @export get_est
get_est <- function(object, v, x = NULL) {
  np <- reticulate::import("numpy")
  if (is.null(x)){
    v <- np$array(v)
    predictions <- object$getCATE(v)
  } else {
    v <- np$array(v)
    x <- np$array(x)
    predictions <- object$predict(x, v)
  }
  predictions
}
