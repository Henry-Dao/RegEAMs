%% Settings
clear all; clc
T_pred = 20; % number of predictive data

%% Save the MCMC/VB results to the format the convenient in R
load('ERP_data.mat')
scaling_data = true;  scaled_std = 1;
if scaling_data == true
    data = scale_data(data,scaled_std);
end

load('MCMC_result.mat')
T_burn = 10001;  T_end = size(MCMC_draws.mu_alpha,2);
thinning_size = floor((T_end - T_burn)/(T_pred-1));
alpha = MCMC_draws.alpha(:,:,T_burn:thinning_size:T_end);
beta_vec = MCMC_draws.beta_vec(:,T_burn:thinning_size:T_end);

save('MCMC_draws_Matlab_to_R.mat','data','alpha','beta_vec','T_pred')

load('VB_result.mat')
lambda_best = VB_results.lambda;
mu = lambda_best.mu; B = lambda_best.B;   d = lambda_best.d;

r = VB_settings.r;

D_alpha = sum(model.subject_param_dim);
num_covariates = model.beta_dim(2);
d_alpha = model.beta_dim(1);

p = D_alpha*(J + 1) + d_alpha*num_covariates;
alpha = zeros(D_alpha,J,T_pred);
beta_vec = zeros(d_alpha*num_covariates,T_pred);
for i = 1:T_pred
    epsilon = randn(p,1);
    z = randn(r,1);
    theta_1 = mu + B*z + d.*epsilon; % theta_1 = (alpha_1,...,alpha_J,mu_alpha,log a_1,...,log a_D)
    alpha(:,:,i) = reshape(theta_1(1:D_alpha*J),D_alpha,J);
    beta_vec(:,i) = theta_1(D_alpha*J + 1:(D_alpha*J+d_alpha*num_covariates));
end
save('VB_draws_Matlab_to_R.mat','data','alpha','beta_vec','T_pred')

