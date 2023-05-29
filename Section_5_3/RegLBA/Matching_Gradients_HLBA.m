function grad_alpha_j = Matching_Gradients_HLBA(MLE_model,data_subject_j,pdf_j,alpha_j_r)   
% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com
%%     
% ---------- Transform random effects to the natural form ---------
    
    c_blk0 = exp(alpha_j_r(:,1));
    c_blk2 = exp(alpha_j_r(:,8));

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
end
