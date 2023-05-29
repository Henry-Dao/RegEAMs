function [f,g] = minus_log_hlba_withbeta_density(x,subject_j,MLE_model)
% INPUT: x = [alpha_j, beta_vec];
% OUTPUT: f = - log pdf
%         g = [grad_alpha_j grad_beta_vec]
% alpha_j ~ N(0,10*I) , beta_vec ~ N(0,10*I)

D_alpha = MLE_model.subject_param_dim;
alpha_j = x(1:D_alpha);
beta_vec = x(1+ D_alpha:end);

omega_ij = MLE_model.matching_parameters(MLE_model,subject_j,alpha_j,beta_vec);
pdf = MLE_model.density(MLE_model,subject_j,omega_ij,true);

covmat_alpha=eye(D_alpha);
covmat_beta=eye(prod(MLE_model.beta_dim));
log_pdf_alpha_j = log(mvnpdf(alpha_j,zeros(1,D_alpha),covmat_alpha));
log_pdf_beta_vec = log(mvnpdf(beta_vec,zeros(1,prod(MLE_model.beta_dim)),covmat_beta));
f = -(pdf.log + log_pdf_alpha_j + log_pdf_beta_vec);
if nargout > 1 
    [grad_alpha_j, grad_beta_j_matrix]  = MLE_model.matching_gradients(MLE_model,subject_j,pdf,alpha_j,beta_vec);
    g = [-(grad_alpha_j'-(covmat_alpha\alpha_j'));...
        -(grad_beta_j_matrix(:)-(covmat_beta\beta_vec'))];
end
end