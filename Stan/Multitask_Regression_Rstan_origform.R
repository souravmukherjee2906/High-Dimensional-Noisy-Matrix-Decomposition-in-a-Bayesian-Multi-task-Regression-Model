library(rstan)
library(truncnorm)
library(rstiefel)
library(invgamma)
library(pracma)
library(psych)
library(mccr)
library(caret)
library(coda)                     
library(e1071)                    
library(ggplot2)
library(tictoc)

q1 <- 0.95  # For any n, Actual data Y_(n*n) (in particular, S_star_(n*n)) will be generated according to q = 0.95
n <- p <- 50
# r <- round(0.05*n)     # rank of the low-rank component is r
r <- 1   # 1 is the rank of the true low-rank matrix L_Star.


## Other input values
a <- 4  # shape parameter for the inverse gamma distribution of sigma^2  ( should be > 0)
b <- 5  # rate parameter for the inverse gamma distribution of sigma^2  ( should be > 0)
tow2 <- 20   # the variance tow^2 of the normal distribution in the mixture prior of S. For the time-being, it's value is taken to be 10.
tow02 <- 10^(-4)


## Getting E_star where each entry is iid N(0, True_Sigma^2)
Sigma2_star <- 0.01
E_star <- matrix(rnorm(n*p, mean = 0, sd = sqrt(Sigma2_star)), nrow = n, ncol = p)

## Getting S_star (according to q1 = 0.95)
S_star_rnorm <- matrix(rnorm(n*p, mean = 0, sd = sqrt(tow2)), nrow = n, ncol = p)
S_star_runif <- matrix(runif(n*p), nrow = n, ncol = p)
S_actual_MCC <- (S_star_runif > q1)*1
S_star <- S_star_rnorm * S_actual_MCC

## Getting U_star of order n*r
## generate a random orthonormal matrix of order n*n. The randomness is meant w.r.t (additively invariant) Haar measure on O(n).
# U_star <- randortho(n, type = "orthonormal")[ ,1:r]                       #takes the first r many columns

## Getting V_star of order p*r
# V_star <- randortho(p, type = "orthonormal")[ ,1:r]

## Getting D_star of order r*r
# d_star <- c(runif(r-1, min = 1, max = 2), runif(1, min = 0.5, max = 1.5))
# D_star <- diag(cumprod(d_star[r:1])[r:1])

# L_star <- U_star %*% D_star %*% t(V_star)
L_star <- (1/n)* (rep(1,n)%*%t(rep(1,p)))

B_star <- L_star + S_star   # each column of B_star represents a task and each row of B_star as a feature.

X <- diag(1, nrow = n, ncol = p) + ((1/sqrt(n)) * (c(1,rep(0,n-1))%*% t(rep(1,p))))   # Design Matrix

## True Data
Y <- (X%*%B_star) + E_star


Simulation.from.stan <- function(n, p, r, q, tow02, tow2, a, b, Y){
  
  ## Compile the model
  Simul_model <- stan_model('Multitask_Regression_Rstan_origform.stan')
  
  ## Initialize the parameters
  init_fn <- function(i){
    X_10 <- matrix(rnorm(n*r), nrow = n, ncol = r)
    X_20 <- matrix(rnorm(p*r), nrow = p, ncol = r)
    d0 <- c(rtruncnorm(r-1, a=1, b=Inf, mean=0, sd=1), rtruncnorm(1, a=0, b=Inf, mean=0, sd=1))    # prior of d
    D0 <- diag(cumprod(d0[r:1])[r:1], nrow = r, ncol = r)
    S_rnorm <- matrix(rnorm(n*p, mean = 0, sd = sqrt(tow2)), nrow = n, ncol = p)
    S_runif <- matrix(runif(n*p), nrow = n, ncol = p)
    S_MCC <- (S_runif > q1)*1
    S0 <- S_rnorm * S_MCC
    sigma20 <- rinvgamma(1, shape = a, rate = b)
    U0 <- svd(X_10)$u %*% t(svd(X_10)$v)
    V0 <- svd(X_20)$u %*% t(svd(X_20)$v)
    S_U0 <- t(X_10)%*%X_10
    S_V0 <- t(X_20)%*%X_20
    L0 <- U0 %*% D0 %*% t(V0)
    return(list(X_1 = X_10, X_2 = X_20, d = d0, S = S0, sigma2 = sigma20, U = U0, V = V0, S_U = S_U0, S_V = S_V0, L = L0))
  }
  
  chains_vec <- 1:4
  init_l <- lapply(chains_vec, function(i) init_fn(i))
  
  ## Pass data to stan and run the model
  #options(mc.cores = 4)
  tic()
  Simul_fit <- sampling(Simul_model, data = list(n=n, p=p, r=r, q = q1, tow02=tow02, tow2=tow2, a=a, b=b, Y=Y, X=X), pars = c("d","L","S","sigma2"), chains = 4, iter = 2000, warmup = 1000, thin = 1, init = init_l, include = TRUE, cores = getOption("mc.cores", 4L), save_warmup = FALSE)
  toc()
  
  ## Diagnose
  print("Simul_fit")
  print(Simul_fit)
  
  ## Extracting parameters and plotting graph
  Simul_params <- extract(Simul_fit, pars = c("d","L","S","sigma2"), permuted = TRUE, inc_warmup = FALSE, include = TRUE)
  print(str(Simul_params))
  d_array <- Simul_params$d
  S_array <- Simul_params$S
  L_array <- Simul_params$L
  Sigma2_vec <- Simul_params$sigma2
  
  
  d_bar <- apply(d_array, 2, FUN = mean)
  print("d_bar")
  print(d_bar)
  #distance_d = max(abs(d_bar - d_star))
  
  # cum_dbar_array <- apply(d_array, 2, FUN = cumsum) * (1/(1:dim(d_array)[1]))
  # pdf('traceplot of cumulative d_bar for 500.pdf', width = 11.694, height = 8.264)
  # trace_d_bar <- traceplot(Simul_fit, pars = "d", include = TRUE, unconstrain = FALSE, inc_warmup = FALSE)  # More parameters are present in traceplot function. 
  # print(trace_d_bar + theme_gray() + coord_cartesian(ylim = c(0, 5)))
  # dev.off()
  
  J <- matrix(1, nrow = n, ncol = p)       # matrix whose all elements are 1
  S_sum <- apply(S_array, c(2,3), FUN = sum)
  S_count_array <- apply(S_array, c(2,3), FUN = function(S) {(S==0)*1})
  S_count <- apply(S_count_array, c(2,3), FUN = sum)
  avg_S <- S_sum / ((dim(S_array)[1]*J) - S_count)
  avg_S[!is.finite(avg_S)] <- 0   # If for any elemnet of avg_S, denom is 0, then NA is replaced by 0, since, in that case, S_count[i,j] = (K+1), which is >= (K+1)/2.
  S_predicted_MCC <- (S_count < (dim(S_array)[1]/2))*1
  S_hat <- avg_S * S_predicted_MCC
  
  L_hat <- apply(L_array, c(2,3), FUN = mean)
  
  B_hat <- L_hat + S_hat
  
  Sigma2_hat <- mean(Sigma2_vec)
  
  
  
  print(c("n is :", n))
  print(c("p is :", p))
  print(c("r is :", r))
  
  ## Finding sensitivity and specificity for all Methods:
  S_actual_factor <- factor(as.factor(S_actual_MCC), levels = c("0", "1"))     ## for the Actual data.
  S_predic_factor <- factor(as.factor(S_predicted_MCC), levels = c("0", "1"))  ## for Method 1: q = 0.95
  print("Sensitivity and Specificity for Method 1: q = 0.95")
  print(sensitivity(S_predic_factor, S_actual_factor, positive = "1"))
  print(specificity(S_predic_factor, S_actual_factor, negative = "0"))
  
  ## Mattews Correlation Coefficient for all Methods:
  print("Mattews Correlation Coefficient(MCC) for S for Method 1: q = 0.95")
  print(mccr(S_actual_MCC, S_predicted_MCC))        # 0 is taken as 0(negative) and non-zero values are taken as 1 (positive)
  
  ## Finding Relative Ratio for L and S for all Methods:
  print("Relative Ratio for L and S for Method 1: q = 0.95")
  print(norm(L_star - L_hat, type = "F")/ norm(L_star, type = "F"))  # Frobenius norm distance
  print(norm(S_star - S_hat, type = "F")/ norm(S_star, type = "F"))  # Frobenius norm distance
  print("Relative Ratio for B and XB for Method 1: q = 0.95")
  print(norm(B_star - B_hat, type = "F")/ norm(B_star, type = "F"))  # Frobenius norm distance
  print(norm((X%*%B_star) - (X%*%B_hat), type = "F")/ norm((X%*%B_star), type = "F"))  # Frobenius norm distance
  
  ## Maximum modulus of all the elements in (L_star - L_hat) for all Methods:
  print("Maximum modulus of all the elements in (L_star - L_hat) for Method 1: q = 0.95")
  print(max(abs(L_star - L_hat)))
  
  ## Maximum modulus of all the elements in (S_star - S_hat) for all Methods:
  print("Maximum modulus of all the elements in (S_star - S_hat) for Method 1: q = 0.95")
  print(max(abs(S_star - S_hat)))
  
  
  print("Absolute distance between Sigma2_bar2 and 0.01 at the last iteration for Method 2")
  print(abs(Sigma2_hat - Sigma2_star))
  
}

