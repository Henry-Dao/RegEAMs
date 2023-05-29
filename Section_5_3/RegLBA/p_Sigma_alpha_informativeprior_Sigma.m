function output = p_Sigma_alpha_informativeprior_Sigma(Sigma_alpha,v,Psi,alpha,mu_alpha,num_covariates,d_alpha)

% INPUT: Sigma_alpha = group-level covariance matrix
%        v = degress of freedom
%        v_a = group-level mean 
%        Psi = scale matrix
%        alpha = random effects (matrix)
%        mu_alpha = group-level mean
%        a = hyperparameter from the marginally non-informative prior of

% OUTPUT: is a structure that contains several fields
%     output.log = log of the density;
%     output.gradient = gradients of log density

% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com

    [p,~] = size(Psi);
    Psi_inv = Psi\eye(p);   Sigma_alpha_inv = Sigma_alpha\eye(p);

    grad_logdet_alpha = 2*Psi_inv*(alpha-mu_alpha);
    grad_logdet_mu_alpha = -sum(grad_logdet_alpha,2); 


    grad_trace_alpha = 2*Sigma_alpha_inv*(alpha-mu_alpha);
    grad_trace_mu_alpha = -sum(grad_trace_alpha,2); 

    
    grad_alpha = 0.5*v*grad_logdet_alpha - 0.5*grad_trace_alpha;
    grad_mu_alpha = 0.5*v*grad_logdet_mu_alpha - 0.5*grad_trace_mu_alpha;


    output.log = 0.5*v*logdet(Psi) - 0.5*v*p*log(2) - log_multigamma(v/2,p) -...
        0.5*(v+p+1)*logdet(Sigma_alpha) - 0.5*trace(Psi*Sigma_alpha_inv);
    output.grad = [grad_alpha(:); zeros(d_alpha*num_covariates,1); grad_mu_alpha];

end