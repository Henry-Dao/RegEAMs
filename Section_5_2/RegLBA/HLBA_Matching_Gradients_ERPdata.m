function [grad_alpha_j, grad_beta_matrix_j] = HLBA_Matching_Gradients_ERPdata(model,subject_j,pdf_j,alpha_j,beta_vec)

% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com
%%     
    R_j = subject_j.R; % (1 = same, 2 = mirror)

    S_j = subject_j.S; % (1 = same, 2 = mirror)  

% --------- Gradients of alpha_c = log(b-A) ----------

    grad_b = sum(pdf_j.grad(:,1:2),2);
    grad_alpha_c = sum(grad_b)*exp(alpha_j(1));

% ----------- Gradients of alpha_A = log(A) -----------------
    
    grad_A = sum(pdf_j.grad(:,3:4),2);  
    grad_alpha_A = sum(grad_b + grad_A)*exp(alpha_j(2));

% --------- Gradients of alpha_tau0 = log(tau0) and alpha_tau = log(tau) ----------
    Rotation_angles = double(subject_j.E*45 -45);
    grad_tau = sum(pdf_j.grad(:,9:10),2);
    grad_alpha_tau0 = sum(grad_tau)*exp(alpha_j(9));
    grad_alpha_tau = sum(grad_tau.*Rotation_angles)*exp(alpha_j(10));

% ---------------- Gradients of drift rate v ~ S*E -----------------

    grad_v_obs_resp = pdf_j.grad(:,5); 
    grad_v_unobs_resp = pdf_j.grad(:,6); 
    grad_v_ij_same = grad_v_obs_resp.*(R_j == 1) + grad_v_unobs_resp.*(R_j == 2);
    grad_v_ij_mirror = grad_v_obs_resp.*(R_j == 2) + grad_v_unobs_resp.*(R_j == 1);
% ----------- Gradients of alpha_v_ss = v_ss -----------------
    grad_v_ss = sum(grad_v_ij_same.*(S_j == 1));
    grad_v_sm = sum(grad_v_ij_same.*(S_j == 2));

    grad_v_ms = sum(grad_v_ij_mirror.*(S_j == 1));
    grad_v_mm = sum(grad_v_ij_mirror.*(S_j == 2));

    grad_v_c = sum(grad_v_ij_same.*(S_j == 1).*Rotation_angles + ...
               grad_v_ij_mirror.*(S_j == 2).*Rotation_angles);

    grad_v_e = sum(grad_v_ij_same.*(S_j == 2).*Rotation_angles + ...
               grad_v_ij_mirror.*(S_j == 1).*Rotation_angles);


    grad_alpha_j = [grad_alpha_c grad_alpha_A grad_v_ss grad_v_sm grad_v_ms ...
        grad_v_mm grad_v_c grad_v_e grad_alpha_tau0 grad_alpha_tau];

grad_beta_matrix_j = sum((grad_v_ij_same + grad_v_ij_mirror).*subject_j.X);

end
