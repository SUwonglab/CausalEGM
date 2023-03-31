#' Install the python CausalEGM package
#'
#' @param method default "auto"
#' @param pip boolean flag, default TRUE
#' @return No return value
#' @importFrom reticulate py_install py_module_available
#' @export install_causalegm
install_causalegm <- function(method = "auto",pip = TRUE) {
  if (!(reticulate::py_module_available('CausalEGM'))) {
    reticulate::py_install("CausalEGM", method = method, pip = pip)
  }
}

#' @title Main function for estimating causal effect in either binary or continuous treatment settings.
#'
#' @description This function takes observation data (x,y,v) as input, and estimate the ATE/ITE/ADRF.
#' @references Qiao Liu, Zhongren Chen, Wing Hung Wong.  CausalEGM: a general causal inference framework by encoding generative modeling. \emph{arXiv preprint arXiv:2212.05925, 2022}.
#' @param x is the treatment variable, one-dimensional array with size n.
#' @param y is the potential outcome, one-dimensional array with size n.
#' @param v is the covariates, two-dimensional array with size n by p.
#' @param z_dims is the latent dimensions for \eqn{z_0,z_1,z_2,z_3} respectively. Total dimension should be much smaller than the dimension of covariates \eqn{v}. Default: c(3,3,6,6)
#' @param output_dir is the folder to save the results including model hyperparameters and the estimated causal effect. Default is ".".
#' @param dataset is the name for the input data. Default: "myData".
#' @param lr is the learning rate. Default: 0.0002.
#' @param bs is the batch size. Default: 32.
#' @param alpha is the coefficient for the reconstruction loss. Default: 1.
#' @param beta is the coefficient for the MSE loss of \eqn{x} and \eqn{y}. Default: 1.
#' @param gamma is the coefficient for the gradient penalty loss. Default: 10.
#' @param g_d_freq is the iteration frequency between training generator and discriminator in the Roundtrip framework. Default: 5.
#' @param g_units is the list of hidden nodes in the generator/decoder network. Default: c(64,64,64,64,64).
#' @param e_units is the list of hidden nodes in the encoder network. Default: c(64,64,64,64,64).
#' @param f_units is the list of hidden nodes in the f network for predicting \eqn{y}. Default: c(64,32,8).
#' @param h_units is the list of hidden nodes in the h network for predicting \eqn{x}. Default: c(64,32,8).
#' @param dv_units is the list of hidden nodes in the discriminator for distribution match \eqn{v}. Default: c(64,32,8).
#' @param dz_units is the list of hidden nodes in the discriminator for distribution match \eqn{z}. Default: c(64,32,8).
#' @param save_res whether to save the results during training. Default: FALSE.
#' @param save_model whether to save the trained model. Default: FALSE.
#' @param binary_treatment whether the treatment is binary or continuous. Default: TRUE.
#' @param use_z_rec whether to use the reconstruction loss for \eqn{z}. Default: TRUE.
#' @param use_v_gan whether to use the GAN training for \eqn{v}. Default: TRUE.
#' @param random_seed is the random seed to fix randomness. Default: 123.
#' @param n_iter is the training iterations. Default: 30000.
#' @param normalize whether apply normalization to covariates. Default: FALSE.
#' @param x_min ADRF start value. Default: NULL
#' @param x_max ADRF end value. Default: NULL
#' @returns \code{causalegm} returns an object of \code{\link[base:class]{class}} "causalegm".
#'
#' An object of class \code{"causalegm"} is a list containing the following:
#'
#'  \item{causal_pre}{the predicted causal effects, which are individual causal effects (ITEs) in binary treatment settings and dose-response values in continous treatment settings.}
#'  \item{getCATE}{the method for getting the conditional average treatment effect (CATE).It takes covariates v as input.}
#'  \item{predict}{the method for outcome function. It takes treatment x and covariates v as inputs.}
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
#' paste("The average treatment effect (ATE):", round(model$ATE, 2))
#' }
#'
#' @importFrom reticulate py_module_available
#' @export causalegm
causalegm <- function(x, y, v,
                        z_dims = c(3,3,6,6),
                        output_dir = ".",
                        dataset = "myData",
                        lr = 0.0002,
                        bs = 32,
                        alpha =1,
                        beta = 1,
                        gamma = 10,
                        g_d_freq = 5,
                        g_units = c(64,64,64,64,64),
                        e_units = c(64,64,64,64,64),
                        f_units = c(64,32,8),
                        h_units = c(64,32,8),
                        dv_units = c(64,32,8),
                        dz_units = c(64,32,8),
                        save_model = FALSE,
                        save_res = FALSE,
                        binary_treatment = TRUE,
                        use_z_rec = TRUE,
                        use_v_gan = TRUE,
                        random_seed = 123,
                        n_iter = 30000,
                        normalize = FALSE,
                        x_min=NULL,
                        x_max=NULL) {

  if (!(reticulate::py_module_available('CausalEGM'))){
    stop("Please install the CausalEGM package using the function: install_causalegm()")
  }
  x <- array(data=x, dim=c(length(x),1))
  y <- array(data=y, dim=c(length(y),1))
  v_dim <- dim(v)[2]
  params <- list(output_dir = output_dir,
                 dataset = dataset,
                 z_dims = as.integer(z_dims),
                 v_dim = v_dim,
                 lr = lr,
                 alpha = alpha,
                 beta = beta,
                 gamma = gamma,
                 g_d_freq = as.integer(g_d_freq),
                 g_units = g_units,
                 e_units = e_units,
                 f_units = f_units,
                 h_units = h_units,
                 dv_units = dv_units,
                 dz_units = dz_units,
                 save_model = save_model,
                 save_res = save_res,
                 binary_treatment = binary_treatment,
                 use_z_rec = use_z_rec,
                 use_v_gan = use_v_gan,
                 x_min = x_min,
                 x_max = x_max)

  cegm <- reticulate::import("CausalEGM")
  model <- cegm$CausalEGM(params=params,random_seed=as.integer(random_seed))
  data <- list(x,y,v)
  model$train(data,
              n_iter = as.integer(n_iter),
              batch_size = as.integer(bs),
              normalize = normalize)

  output <- list(
    "causal_pre" = model$best_causal_pre, "getCATE" = model$getCATE,
    "predict" = model$predict)

  class(output) <- "causalegm"
  return(output)
}
