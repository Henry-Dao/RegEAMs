function [f,g] = minus_log_hddm_density(x,subject_j,MLE_model)

alpha_j = x;  
D_alpha = length(alpha_j);
z_ij = MLE_model.matching_parameters(MLE_model,subject_j,alpha_j);
pdf = MLE_model.density(MLE_model,subject_j,z_ij,true);
covmat = eye(D_alpha);
f = -(pdf.log+log(mvnpdf(alpha_j,zeros(1,D_alpha),covmat)));
if nargout > 1 
    grad_alpha_j = MLE_model.matching_gradients(MLE_model,subject_j,pdf,alpha_j);
    g = -(grad_alpha_j-(covmat\alpha_j'));
end
end