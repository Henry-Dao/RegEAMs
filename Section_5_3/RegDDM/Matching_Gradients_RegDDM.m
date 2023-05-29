function [grad_alpha_j, grad_beta_matrix_j] = Matching_Gradients_HDDM_HCPdata_model1(model,subject_j,pdf_j,alpha_j_r,beta_vec)

%% Input: alpha_j_r is a particle (row vector), not a matrix
% alpha_j_r = ( log(-v0_lure), log(v0_target), log(-v0_nontarget), log(sv0), 
%               log(a0-z0-sz0/2), log(z0-sz0/2), log(sz0), log(tau0 - stau0/2), log(stau0),
%               log(-v2_lure), log(v2_target), log(-v2_nontarget), log(sv2), 
%               log(a2-z2-sz2/2), log(z2-sz2/2), log(sz2), log(tau2 - stau2/2), log(stau2))
% beta_matrix = [beta_v0_lure; beta_v0_target; beta_v0_nontarget; beta_a0; 
%                beta_v2_lure; beta_v2_target; beta_v2_nontarget; beta_a2] 
%% linking equation: 
%             log(-v) = log(-v0_lure) + beta_v0_lure*X_j, if block = 0 and Cond =  lure
%             log(v) = log(v0_lure) + beta_v0_target*X_j, if block = 0 and Cond =  target
%             log(-v) = log(-v0_nontarget) + beta_v0_nontarget*X_j, if block = 0 and Cond =  nontarget
%             log(a-z-sz/2) = log(a0-z0-sz0/2) + beta_a0*X_j, if block = 0

%             log(-v) = log(-v2_lure) + beta_v2_lure*X_j, if block = 2 and Cond =  lure
%             log(v) = log(v2_lure) + beta_v2_target*X_j, if block = 2 and Cond =  target
%             log(-v) = log(-v2_nontarget) + beta_v2_nontarget*X_j, if block = 2 and Cond =  nontarget
%             log(a-z-sz/2) = log(a2-z2-sz2/2) + beta_a2*X_j, if block = 2
%% Data:
% data is a cell array, each cell corresponds to a subject. Each cell is a
% structure having 5 fieds
%     - RT = observed response time (seconds)
%     - RE = observed response (1 = target, 0 = nontarget)
%     - Blk = Block type (1 = 2-back, 0 = 0-back)
%     - E = experimental conditions (1 = lure, 2 = target, 3 = nontarget)
%     - X = a row vector of covariates
    
% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com
%%     
% ---------- Transform random effects to the natural form ---------
    beta_matrix = reshape(beta_vec,model.beta_dim(1),model.beta_dim(2));
    betaX_j = beta_matrix*subject_j.X';
    idx_linked_components = [1:3, 5, 10:12, 14];
    alpha_j_r(idx_linked_components) = alpha_j_r(idx_linked_components) + betaX_j'; 

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
    grad_alpha_j(1,i) = sum(grad_alpha_tilde_j(idx,i),1,'omitnan'); % acutually using isfinite() we no longer need 'omitnan'
end

idx = sum(isnan(grad_alpha_tilde_j),2) ==0;
grad_beta_matrix_j = sum(grad_alpha_tilde_j(idx,idx_linked_components))'.*subject_j.X;

end
