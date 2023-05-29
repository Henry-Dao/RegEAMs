function [f,g] = minus_log_hddm_nobeta_density(x,subject_j,MLE_model)

alpha_j = x(1:18);  
z_ij = MLE_model.matching_parameters(MLE_model,subject_j,alpha_j);
pdf = MLE_model.density(MLE_model,subject_j,z_ij,true);
covmat=eye(18);
f = -(pdf.log+log(mvnpdf(alpha_j,zeros(1,18),covmat)));
if nargout > 1 
    grad_alpha_j = MLE_model.matching_gradients(MLE_model,subject_j,pdf,alpha_j);
    g = -(grad_alpha_j'-(covmat\alpha_j'));
end
end