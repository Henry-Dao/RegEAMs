function output = prior_density_Hybrid_informativeprior_Sigma(model,beta_vec,alpha,mu_alpha,Sigma_alpha)

% INPUT: model = structure that contains model specifications
%        alpha = a matrix of transform random effects
%        mu_alpha = group-level mean 
%        Sigma_alpha = group-level covariance matrix
%        a = hyperparameter from the marginally non-informative prior of
%        the covariance matrix Sigma_alpha


% OUTPUT: is a structure that contains several fields
%     output.log = log of the prior density;
%     output.gradient = gradients of log prior density wrt alpha

% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com

    J = model.num_subjects; 

%% Transform theta to C,Lambda and Sigma_alpha

    D_alpha = sum(model.subject_param_dim); % number of random effects per participant
    v = model.prior_par.v;
    mu_beta = model.prior_par.mu_beta;
    Sig_beta = model.prior_par.Sigma_beta;
    Sigma_alpha_inv = Sigma_alpha\eye(D_alpha);

%% Compute gradients

% ------------- gradient wrt to alpha_1,...,alpha_n ----------------------
    A = Sigma_alpha_inv*(alpha - mu_alpha); 
    grad_alpha  = -A(:);
    
% ------------- gradient wrt to beta_vec ----------------------    

    temp = -Sig_beta\(beta_vec - mu_beta);
    grad_beta_vec  = temp(:);
%---------------------- gradient wrt mu_alphaA -------------------------

    grad_mu = sum(A,2) - model.prior_par.cov\(mu_alpha-model.prior_par.mu); 

    output.log = -0.5*D_alpha*J*log(2*pi) - 0.5*J*logdet(Sigma_alpha)...
        -0.5*trace((alpha-mu_alpha)'*A) -0.5*D_alpha*J*log(2*pi) - 0.5*J*logdet(Sigma_alpha)...
        -0.5*trace((alpha-mu_alpha)'*A) -0.5*D_alpha*log(2*pi) - 0.5*logdet(model.prior_par.cov)...
        -0.5*(mu_alpha-model.prior_par.mu)'/model.prior_par.cov*(mu_alpha-model.prior_par.mu) + 0.5*v*logdet(model.prior_par.Psi) - 0.5*v*D_alpha*log(2)...
        -log_multigamma(v/2,D_alpha)- 0.5*(model.prior_par.v + D_alpha + 1)*logdet(Sigma_alpha)...
        -0.5*trace(model.prior_par.Psi*Sigma_alpha_inv) + log(mvnpdf(beta_vec',mu_beta',Sig_beta));
    output.grad = [grad_alpha; grad_beta_vec; grad_mu]; % gradient of log(theta) wrt theta
end
