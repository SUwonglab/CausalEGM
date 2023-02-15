#' @title Main function for CausalEGM to estimate causal effect in either binary or continuous treatment settings.
#'
#' @description This function takes observation data (x,y,v) as input, and estimate the ATE/ITE/ADRF.
#'
#' @param X is the treatment variable.
#' @param Y is the potential outcome.
#' @param V is the covariates.
#'
#'
#' @return NULL
#'
#' @examples causalegm(X=X,Y=Y,V=V,yaml_file='example.yaml')
#'
#' @export causalegm
causalegm <- function(X, Y, V,
                        output_dir = "./",
                        dataset = "myData",
                        z_dims = c(3,3,6,6),
                        lr = 0.0002,
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
                        binary_treatment = TRUE,
                        use_z_rec = TRUE,
                        use_v_gan = TRUE,
                        random_seed = 123,
                        n_iter = 20000) {

  #To ignore the warnings during usage
  options(warn=-1)
  options("getSymbols.warning4.0"=FALSE)
  if (!(py_module_available('CausalEGM'))){
    py_install("CausalEGM", method="auto",pip=TRUE)
  }
  #get number of covariates
  v_dim <- dim(V)[2]
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
                 binary_treatment = binary_treatment,
                 use_z_rec = use_z_rec,
                 use_v_gan = use_v_gan)

  cegm <- import("CausalEGM")
  model <- cegm$CausalEGM(params=params,random_seed=as.integer(random_seed))
  data <- list(X,Y,V)
  model$train(data,n_iter = as.integer(n_iter))
  model
}
