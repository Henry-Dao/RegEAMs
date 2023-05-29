function [grad_alpha_j, grad_beta_matrix_j] = Matching_Gradients_HLBA_HCPdata_model2(model,data_subject_j,pdf_j,alpha_j_r,beta_vec)

%% Input: alpha_j_r is a particle (row vector), not a matrix
% alpha_j_r = [log(c0), log(A0), log(v0), log(v0_lure), log(v0_target), log(v0_nontarget), log(tau0),
%              log(c2), log(A2), log(v2), log(v2_lure), log(v2_target), log(v2_nontarget), log(tau2)]
% beta_matrix = [log(c0); v_blk0_lure_target; v_blk0_lure_nontarget; v_blk0_target_target; v_blk0_target_nontarget; v_blk0_nontarget_target; v_blk0_nontarget_nontarget;
%                log(c2); v_blk2_lure_target; v_blk2_lure_nontarget; v_blk2_target_target; v_blk2_target_nontarget; v_blk2_nontarget_target; v_blk2_nontarget_nontarget] 
%% linking equation: 
%             log(c_ij) = log(c_blk) + beta_c*X_j
%             v_target_ij = v_blk +/- v_blk_cond + beta_blk_cond_target*X_j
%             v_nontarget_ij = v_blk +/- v_blk_cond + beta_blk_cond_nontarget*X_j
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
    betaX_j = beta_matrix*data_subject_j.X';
    c_blk0 = exp(alpha_j_r(:,1) + betaX_j(1,:)');
    c_blk2 = exp(alpha_j_r(:,8) + betaX_j(8,:)');

    A_blk0 =  exp(alpha_j_r(:,2));  A_blk2 =  exp(alpha_j_r(:,2+7));
    tau_blk0 =  exp(alpha_j_r(:,7));  tau_blk2 =  exp(alpha_j_r(:,7+7));

%% Gradient wrt alpha_ij   
    Block_j = data_subject_j.Blk;
    E_j = data_subject_j.E; % (1 = lure, 2 = target, 3 = nontarget)
    
    cond1_idx = (E_j == 1); cond2_idx = (E_j == 2); cond3_idx = (E_j == 3);
    blk0_idx = (Block_j == 0);      blk2_idx = (Block_j == 1);

% --------------- Extract gradients wrt eta_ij --------------------   
    
    grad_b = sum(pdf_j.grad(:,1:2),2);
    grad_A = sum(pdf_j.grad(:,3:4),2);
    grad_tau = sum(pdf_j.grad(:,9:10),2);
    R_j = data_subject_j.RE;
    grad_v_target = pdf_j.grad(:,5).*(R_j == 1) + pdf_j.grad(:,6).*(R_j == 0);    
    grad_v_nontarget = pdf_j.grad(:,5).*(R_j == 0) + pdf_j.grad(:,6).*(R_j == 1); 

    
% --------------- Gradients wrt alpha_j --------------------  
% alpha_j_r = [log(c0), log(A0), log(v0), log(v0_lure), log(v0_target), log(v0_nontarget), log(tau0),
%              log(c2), log(A2), log(v2), log(v2_lure), log(v2_target), log(v2_nontarget), log(tau2)]
    
    grad_alpha_tau_blk0 = sum(grad_tau(blk0_idx))*tau_blk0;
    grad_alpha_tau_blk2 = sum(grad_tau(blk2_idx))*tau_blk2;
    grad_alpha_c_blk0 = sum(grad_b(blk0_idx))*c_blk0;
    grad_alpha_c_blk2 = sum(grad_b(blk2_idx))*c_blk2;
    grad_alpha_A_blk0 = sum(grad_b(blk0_idx) + grad_A(blk0_idx))*A_blk0;
    grad_alpha_A_blk2 = sum(grad_b(blk2_idx) + grad_A(blk2_idx))*A_blk2;

    grad_alpha_v_blk0 = (sum(grad_v_target(blk0_idx)) + sum(grad_v_nontarget(blk0_idx)))*exp(alpha_j_r(:,3));   
    grad_alpha_v_blk2 = (sum(grad_v_target(blk2_idx)) + sum(grad_v_nontarget(blk2_idx)))*exp(alpha_j_r(:,10)); 

    grad_alpha_vlure_blk0 = (sum(grad_v_nontarget(blk0_idx&cond1_idx)) - ...
        sum(grad_v_target(blk0_idx&cond1_idx)) )*exp(alpha_j_r(:,4));  
    grad_alpha_vlure_blk2 = (sum(grad_v_nontarget(blk2_idx&cond1_idx)) - ...
        sum(grad_v_target(blk2_idx&cond1_idx)) )*exp(alpha_j_r(:,11));  

    grad_alpha_vtarget_blk0 = (-sum(grad_v_nontarget(blk0_idx&cond2_idx)) + ...
        sum(grad_v_target(blk0_idx&cond2_idx)) )*exp(alpha_j_r(:,5)); 
    grad_alpha_vtarget_blk2 = (-sum(grad_v_nontarget(blk2_idx&cond2_idx)) + ...
        sum(grad_v_target(blk2_idx&cond2_idx)) )*exp(alpha_j_r(:,12));

    grad_alpha_vnontarget_blk0 = (sum(grad_v_nontarget(blk0_idx&cond3_idx)) - ...
        sum(grad_v_target(blk0_idx&cond3_idx)) )*exp(alpha_j_r(:,6)); 
    grad_alpha_vnontarget_blk2 = (sum(grad_v_nontarget(blk2_idx&cond3_idx)) - ...
        sum(grad_v_target(blk2_idx&cond3_idx)) )*exp(alpha_j_r(:,13)); 

grad_alpha_j = [grad_alpha_c_blk0, grad_alpha_A_blk0, grad_alpha_v_blk0, grad_alpha_vlure_blk0,...
                grad_alpha_vtarget_blk0, grad_alpha_vnontarget_blk0, grad_alpha_tau_blk0,... 
                grad_alpha_c_blk2, grad_alpha_A_blk2, grad_alpha_v_blk2, grad_alpha_vlure_blk2,...
                grad_alpha_vtarget_blk2, grad_alpha_vnontarget_blk2, grad_alpha_tau_blk2];


grad_beta_matrix_j = [grad_alpha_c_blk0; sum(grad_v_target(blk0_idx&cond1_idx)); sum(grad_v_nontarget(blk0_idx&cond1_idx)); ...
    sum(grad_v_target(blk0_idx&cond2_idx)); sum(grad_v_nontarget(blk0_idx&cond2_idx));...
    sum(grad_v_target(blk0_idx&cond3_idx)); sum(grad_v_nontarget(blk0_idx&cond3_idx));...
    grad_alpha_c_blk2; sum(grad_v_target(blk2_idx&cond1_idx)); sum(grad_v_nontarget(blk2_idx&cond1_idx)); ...
    sum(grad_v_target(blk2_idx&cond2_idx)); sum(grad_v_nontarget(blk2_idx&cond2_idx));...
    sum(grad_v_target(blk2_idx&cond3_idx)); sum(grad_v_nontarget(blk2_idx&cond3_idx))].*data_subject_j.X; 
end
