function grad_alpha_j = Matching_Gradients_HDDM_Lexical(model,data_subject_j,pdf_j,alpha_j)
% INPUT: model = structure that contains model specifications
%        LBA_j = output of LBA_pdf function (structure)
%        z_j = output of Matching_function (structure)
%        data_subject_j = a structure contains all observations from
%        subject j
% OUTPUT: grad_alpha_j = the partial derivatives wrt the random effect
%        alpha_j

% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com

%% Calculate the gradients with respect to \alpha_j
    W_j = data_subject_j.W;
    I_hf = (W_j == 1); I_lf = (W_j == 2);   I_vlf = (W_j == 3);
    I_nw = (W_j == 4);
  
    grad_v = [pdf_j.grad(:,1).*I_hf, pdf_j.grad(:,1).*I_lf, pdf_j.grad(:,1).*I_vlf , pdf_j.grad(:,1).*I_nw];
    
    E_j = data_subject_j.E;
        
    I_sp = (E_j == 2); I_acc = (E_j == 1); 
    grad_a = [pdf_j.grad(:,3).*I_sp, pdf_j.grad(:,3).*I_acc];
    grad_z = [pdf_j.grad(:,4).*I_sp, pdf_j.grad(:,4).*I_acc];
% --------- Transformation 1 ---------------------
    temp = zeros(data_subject_j.num_trials,12);
    temp(:,1:4) = grad_v;
    temp(:,5) = pdf_j.grad(:,2);
    temp(:,6:7) = grad_a;
    temp(:,8:9) = grad_a + grad_z;
    temp(:,10) = sum(grad_a,2) + 0.5*sum(grad_z,2) + pdf_j.grad(:,5);
    temp(:,11) = pdf_j.grad(:,6);
    temp(:,12) = 0.5*pdf_j.grad(:,6) + pdf_j.grad(:,7);
    
    grad_alpha_j = sum(temp,1)'.*[ones(4,1); exp(alpha_j(5:end)')];  
                                                                           
end