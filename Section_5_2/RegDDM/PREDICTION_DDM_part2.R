library(rtdists)
library(mvtnorm)
library(R.matlab)
library(doParallel)


# --------------------------- load results from Matlab  ---------------------------------

Matlab_data <- readMat('VB_draws_Matlab_to_R.mat')
estimation_method <- 'VB'


# --------------------------- Generate posterior predictive data ---------------------------------
set.seed(2023)
J <- length(Matlab_data$data)
library(doParallel)
cl = makeCluster(4)
registerDoParallel(cl)

XX <- foreach (j = 1:J)%dopar% {
  library(rtdists)
  print(paste0('Simulating predicted data for subject ',j))
  predicted_data <- data.frame()
  i <- 1
  for (t in 1:Matlab_data$T.pred) {
    print(paste0('=============',round(i/Matlab_data$T.pred,2)*100,'% completed '))
    n_j <- unlist(Matlab_data$data[[j]][[1]][5])
    beta_matrix <- matrix(Matlab_data$beta.vec[,t],nrow = 8,byrow = FALSE)
    betaX_j <- matrix(unlist(Matlab_data$data[[j]][[1]][6]),ncol =8)%*%beta_matrix
    alpha_j <- Matlab_data$alpha[,j,t]
    
    v0 <- alpha_j[1] 
    v <- alpha_j[2] 
    sv <- exp(alpha_j[3] )
    sz <- exp(alpha_j[6] )
    z <- exp(alpha_j[5] ) + sz/2
    a <- exp(alpha_j[4] ) + z + sz/2
    stau <- exp(alpha_j[9] )
    tau <- exp(alpha_j[8] )
    tau0 <- exp(alpha_j[7]) + stau/2
    
    rotation <- (unlist(Matlab_data$data[[j]][[1]][3]) - 1)*45
    same_idx <- unlist(Matlab_data$data[[j]][[1]][4]) == 1
    v_ij <- -v0 - rotation*v + betaX_j
    v_ij[same_idx] <- v0 + rotation[same_idx]*v + betaX_j[same_idx]

    tau_ij <- tau0 + rotation*tau
    
    tempt <- rdiffusion( n = n_j, a = a, v = v_ij, t0 = tau_ij - 0.5 * stau,
                         z = z, d = 0, sz = sz, sv = sv, st0 = stau, s = 1)
    num_responses <- rep(1,n_j)
    num_responses[tempt$response == 'lower'] <- 2
    tempt$response <- num_responses
    i <- i+1
    predicted_data <- rbind(predicted_data,tempt)
  }
  predicted_data
}
predicted_data <- XX
stopCluster(cl)


# --------------------------------------- save data  --------------------------------


for (j in 1:J) {
  writeMat(
    con = paste(estimation_method,'_R_to_Matlab_predicted_data_subject_',j,'.mat',sep = ""),
    predicted_data = predicted_data[[j]], T_pred = Matlab_data$T.pred
  )
}

#-------------------------------------------------------------------------------

# --------------------------- load results from Matlab  ---------------------------------

Matlab_data <- readMat('MCMC_draws_Matlab_to_R.mat')
estimation_method <- 'MCMC'


# --------------------------- Generate posterior predictive data ---------------------------------
set.seed(2023)
J <- length(Matlab_data$data)
library(doParallel)
cl = makeCluster(4)
registerDoParallel(cl)

XX <- foreach (j = 1:J)%dopar% {
  library(rtdists)
  print(paste0('Simulating predicted data for subject ',j))
  predicted_data <- data.frame()
  i <- 1
  for (t in 1:Matlab_data$T.pred) {
    print(paste0('=============',round(i/Matlab_data$T.pred,2)*100,'% completed '))
    n_j <- unlist(Matlab_data$data[[j]][[1]][5])
    beta_matrix <- matrix(Matlab_data$beta.vec[,t],nrow = 8,byrow = FALSE)
    betaX_j <- matrix(unlist(Matlab_data$data[[j]][[1]][6]),ncol =8)%*%beta_matrix
    alpha_j <- Matlab_data$alpha[,j,t]
    
    v0 <- alpha_j[1] 
    v <- alpha_j[2] 
    sv <- exp(alpha_j[3] )
    sz <- exp(alpha_j[6] )
    z <- exp(alpha_j[5] ) + sz/2
    a <- exp(alpha_j[4] ) + z + sz/2
    stau <- exp(alpha_j[9] )
    tau <- exp(alpha_j[8] )
    tau0 <- exp(alpha_j[7]) + stau/2
    
    rotation <- (unlist(Matlab_data$data[[j]][[1]][3]) - 1)*45
    same_idx <- unlist(Matlab_data$data[[j]][[1]][4]) == 1
    v_ij <- -v0 - rotation*v + betaX_j
    v_ij[same_idx] <- v0 + rotation[same_idx]*v + betaX_j[same_idx]
    
    tau_ij <- tau0 + rotation*tau
    
    tempt <- rdiffusion( n = n_j, a = a, v = v_ij, t0 = tau_ij - 0.5 * stau,
                         z = z, d = 0, sz = sz, sv = sv, st0 = stau, s = 1)
    num_responses <- rep(1,n_j)
    num_responses[tempt$response == 'lower'] <- 2
    tempt$response <- num_responses
    i <- i+1
    predicted_data <- rbind(predicted_data,tempt)
  }
  predicted_data
}
predicted_data <- XX
stopCluster(cl)


# --------------------------------------- save data  --------------------------------


for (j in 1:J) {
  writeMat(
    con = paste(estimation_method,'_R_to_Matlab_predicted_data_subject_',j,'.mat',sep = ""),
    predicted_data = predicted_data[[j]], T_pred = Matlab_data$T.pred
  )
}