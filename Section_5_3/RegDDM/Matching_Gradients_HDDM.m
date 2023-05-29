function grad_alpha_j = Matching_Gradients_HDDM(MLE_model,subject_j,pdf_j,alpha_j_r)
    
% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com
%%     

%% Gradient wrt alpha_ij   
    Block_j = subject_j.Blk;
    E_j = subject_j.E; % (1 = lure, 2 = target, 3 = nontarget)
 

    cond1_idx = (E_j == 1); cond2_idx = (E_j == 2); cond3_idx = (E_j == 3);
    blk0_idx = (Block_j == 0);      blk2_idx = (Block_j == 1);

% --------------- Extract gradients wrt eta_ij --------------------   

    grad_mu_v = pdf_j.grad(:,1);
    grad_sv = pdf_j.grad(:,2);
    grad_a = pdf_j.grad(:,3);
    grad_z = pdf_j.grad(:,4);
    grad_sz = pdf_j.grad(:,5);
    grad_t0 = pdf_j.grad(:,6);
    grad_st0 = pdf_j.grad(:,7);


% --------------- Match gradients wrt eta_ij with trials --------------------  

    grad_blk0 = [grad_mu_v.*[-cond1_idx, cond2_idx, -cond3_idx], ...
                  grad_sv, grad_a, (grad_a + grad_z), (grad_a + 0.5*grad_z + grad_sz),...
                  grad_t0, (0.5*grad_t0 + grad_st0)].*exp(alpha_j_r(:,1:9)).*blk0_idx;

    grad_blk2 = [grad_mu_v.*[-cond1_idx, cond2_idx, -cond3_idx], ...
                  grad_sv, grad_a, (grad_a + grad_z), (grad_a + 0.5*grad_z + grad_sz),...
                  grad_t0, (0.5*grad_t0 + grad_st0)].*exp(alpha_j_r(:,10:18)).*blk2_idx;

    grad_alpha_tilde_j = [grad_blk0 grad_blk2];


grad_alpha_j = zeros(1,size(grad_alpha_tilde_j,2));
for i = 1:size(grad_alpha_tilde_j,2)
    idx = isfinite(grad_alpha_tilde_j(:,i)); % Discard observations resulting in NaN or Inf gradients
    grad_alpha_j(1,i) = sum(grad_alpha_tilde_j(idx,i),1,'omitnan');
end

end
