#### Simulate linear effect non-linear
linear_effect_non_linear<- function(seed=1,  N = 20000, v_dim=200, z0_dim=2, z1_dim=2, z2_dim=2, z3_dim=4, ax = 1, bx = 1){
  set.seed(seed)
  v <- matrix(rnorm(N*v_dim,mean=0,sd=0.5), N, v_dim) 
  alpha <- rep(1, z0_dim + z1_dim)
  beta_1 <- rep(1, z0_dim)
  beta_2 <- rep(1, z2_dim)
  
  simulate_normal <- function(mu) rnorm(1, mean = mu, sd = 0.5)
  mu_x <- v[, 1:(z0_dim+z1_dim)]%*%alpha
  x <- Map(simulate_normal, mu_x)
  mu_y <- mu_x + (mu_x)**2 * (v[, 1:z0_dim]%*%beta_1 + v[, (z0_dim+z1_dim+1):(z0_dim+z1_dim+z2_dim)]%*%beta_2)
  y <- Map(simulate_normal, mu_y)
  
  return (list(v=v, x=x, y=y))
}

Imbens_Sim <- function(N=20000, V_dim=200, seed=1){
  set.seed(seed)
  V <- matrix(rexp(N*V_dim), N, V_dim) 
  Z0 <- V[,1]
  Z1 <- V[,2]
  Z2 <- V[,3]
  Z3 <- V[,4]
  X <- rexp(N, Z0+Z1)
  gps <- (Z0+Z1) * exp(-(Z0+Z1)*X)
  Y <- rnorm(N, mean=X+(Z0+Z2)*exp(-X*(Z0+Z2)), sd=1)
  # Imbens_sim_data <- data.frame(cbind(V, X, Y, gps))
  Imbens_sim_data <- data.frame(cbind(V, X, Y))
  return(Imbens_sim_data)
}
