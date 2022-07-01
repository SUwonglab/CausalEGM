
#### Simulate linear effect
linear_effect_Sim<- function(seed=1,  N = 20000, v_dim=200, z0_dim=2, z1_dim=2, z2_dim=2, ax = 1, bx = 1){
  set.seed(seed)
  V <- matrix(rnorm(N*v_dim,mean=0,sd=1), N, v_dim) 
  alpha <- rep(1, z0_dim + z1_dim)
  beta_1 <- rep(1, z0_dim)
  beta_2 <- rep(1, z2_dim)
  
  simulate_normal <- function(mu) rnorm(1, mean = mu, sd = 1)
  mu_x <- V[, 1:(z0_dim+z1_dim)]%*%alpha
  X <- unlist(Map(simulate_normal, mu_x))
  mu_y <- X + (X)**2 * (V[, 1:z0_dim]%*%beta_1 + V[, (z0_dim+z1_dim+1):(z0_dim+z1_dim+z2_dim)]%*%beta_2)
  Y <- unlist(Map(simulate_normal, mu_y))
  return(data.frame(cbind(V, X, Y)))
}

#### Simulate nolinear effect
nolinear_effect_Sim<- function(seed=1,  N = 20000, v_dim=200, z0_dim=3, z1_dim=3, z2_dim=3, ax = 1, bx = 1){
  set.seed(seed)
  V <- matrix(rnorm(N*v_dim,mean=0,sd=1), N, v_dim) 
  alpha <- rep(1, z0_dim + z1_dim)
  beta_1 <- rep(1, z0_dim)
  beta_2 <- rep(1, z2_dim)
  
  simulate_normal <- function(mu) rnorm(1, mean = mu, sd = 0.1)
  mu_x <- V[, 1:(z0_dim+z1_dim)]%*%alpha
  X <- unlist(Map(simulate_normal, mu_x))
  mu_y <- (X)**2 + X * (V[, 1:z0_dim]%*%beta_1 + V[, (z0_dim+z1_dim+1):(z0_dim+z1_dim+z2_dim)]%*%beta_2)
  Y <- unlist(Map(simulate_normal, mu_y))
  return(data.frame(cbind(V, X, Y)))
}

#### Simulate  effect from Hirano_Imbens paper
Imbens_Sim <- function(N=20000, v_dim=200, seed=1){
  set.seed(seed)
  V <- matrix(rexp(N*v_dim), N, v_dim) 
  Z0 <- V[,1]
  Z1 <- V[,2]
  Z2 <- V[,3]
  Z3 <- V[,4]
  X <- rexp(N, Z0+Z1)
  gps <- (Z0+Z1) * exp(-(Z0+Z1)*X)
  Y <- rnorm(N, mean=X+(Z0+Z2)*exp(-X*(Z0+Z2)), sd=1)
  return(data.frame(cbind(V, X, Y)))
}