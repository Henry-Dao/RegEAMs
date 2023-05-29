function Psi_j = HLBA_Matching_Parameters_ERPdata(model,subject_j,alpha_j_R,beta_vec)

% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com

%%     
% ---------- Transform random effects to the natural form ---------
    R = size(alpha_j_R,1);
    beta_matrix = reshape(beta_vec,model.beta_dim(1),model.beta_dim(2));
    betaX_j = subject_j.X*beta_matrix';
        
    A = exp(alpha_j_R(:,2));    b = A + exp(alpha_j_R(:,1));
    v_ss = alpha_j_R(:,3);   v_sm = alpha_j_R(:,4);
    v_ms = alpha_j_R(:,5);   v_mm = alpha_j_R(:,6);
    v_c = alpha_j_R(:,7);   v_e = alpha_j_R(:,8);
    tau0 = exp(alpha_j_R(:,9));   tau = exp(alpha_j_R(:,10));
        
% ---------- Identify the parameters at each observation ---------   

        % ----------- drift rate v ~ S*E -----------------
        same_idx_stack = repmat(subject_j.S,R,1)  == 1; % S = 1 (same)
        same_idx = subject_j.S==1;

        n_j = subject_j.num_trials;
        Rotation_angles = double(subject_j.E*45 -45);
        v_ij_same = kron(v_sm,ones(n_j,1)) + kron(v_e,Rotation_angles)   + repmat(betaX_j,R,1);
        v_ij_same(same_idx_stack) = kron(v_ss,ones(sum(same_idx),1)) + kron(v_c,Rotation_angles(same_idx)) + repmat(betaX_j(same_idx,:),R,1);
    
        v_ij_mirror = kron(v_mm,ones(n_j,1)) + kron(v_c,Rotation_angles)   + repmat(betaX_j,R,1);
        v_ij_mirror(same_idx_stack) = kron(v_ms,ones(sum(same_idx),1)) + kron(v_e,Rotation_angles(same_idx)) + repmat(betaX_j(same_idx,:),R,1);
        
        R_j = subject_j.R;
        v_obs_resp = (R_j == 1).*v_ij_same + (R_j == 2).*v_ij_mirror; 
        v_unnobs_resp = (R_j == 1).*v_ij_mirror + (R_j == 2).*v_ij_same; 
        
        % ----------- non-decision tau ~ E -----------------
        tau_ij = kron(tau0,ones(n_j,1)) + kron(tau,Rotation_angles);    

 % ---------- Stack the parameters if alpha_j_R is a matrix ---------   
        b_ij = kron(b,ones(n_j,1));
        A_ij = kron(A,ones(n_j,1));
        sv_ij = ones(R*subject_j.num_trials,1);
        Psi_j = [b_ij b_ij A_ij A_ij v_obs_resp v_unnobs_resp sv_ij sv_ij tau_ij tau_ij];
       
end
