function [f,g] = minus_log_lba_nobeta_density(x,subject_j,MLE_model)
alpha_j = x(1:14); 
z_ij = MLE_model.matching_parameters(MLE_model,subject_j,alpha_j);
pdf = MLE_model.density(MLE_model,subject_j,z_ij,true);
covmat=16*eye(14);
f = -(pdf.log+log(mvnpdf(alpha_j,zeros(1,14),covmat)));
%f = -(pdf.log);
%f = -(pdf.log+log(normpdf(alpha_j(7),0,1))+log(normpdf(alpha_j(14),0,1)));

if nargout > 1 
    grad_alpha_j = MLE_model.matching_gradients(MLE_model,subject_j,pdf,alpha_j);
    g = -(grad_alpha_j'-(covmat\alpha_j'));
   
end
end