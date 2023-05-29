function Psi_j = Matching_Parameters_HDDM_HCPdata_model1(model,subject_j,alpha_j_r,beta_vec)
%% Input: alpha_j_r is a particle (row vector), not a matrix
% alpha_j_r = ( log(-v0_lure), log(v0_target), log(-v0_nontarget), log(sv0), 
%               log(a0-z0-sz0/2), log(z0-sz0/2), log(sz0), log(tau0 - stau0/2), log(stau0),
%               log(-v2_lure), log(v2_target), log(-v2_nontarget), log(sv2), 
%               log(a2-z2-sz2/2), log(z2-sz2/2), log(sz2), log(tau2 - stau2/2), log(stau2))
% beta_matrix = [beta_v0_lure; beta_v0_target; beta_v0_nontarget; beta_a0; 
%                beta_v2_lure; beta_v2_target; beta_v2_nontarget; beta_a2] 
%% linking equation: 
%             log(-v) = log(-v0_lure) + beta_v0_lure*X_j, if block = 0 and Cond =  lure
%             log(v) = log(v0_target) + beta_v0_target*X_j, if block = 0 and Cond =  target
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
    z_j = model.T_inv(alpha_j_r);
    % z_j = ( v^0_lure, v^0_target, v^0_nontarget, sv^0, a^0, z^0, sz^0, tau^0, stau^0,
    %         v^2_lure, v^2_target, v^2_nontarget, sv^2, a^2, z^2, sz^2, tau^2, stau^2)
        
% ---------- Match the random effects with each observation ---------   

        % ----------- drift rate v ~ S*E -----------------
        Block_j = subject_j.Blk;
        E_j = subject_j.E; % (1 = lure, 2 = target, 3 = nontarget)
        mu_v_j = z_j([1:3,10:12]); % mu_v_j = (v0_lure, v0_target, v0_nontarget,v2_lure, v2_target, v2_nontarget)
        v_ij = zeros(subject_j.num_trials,1);  
        for k = 1:2
            for e = 1:3
                v_ij = v_ij + (E_j == e).*(Block_j == k-1)*mu_v_j(3*(k-1)+e);
            end
        end

        % ----------- non-decision tau ~ E -----------------
        % z_j = ( v^0_lure, v^0_target, v^0_nontarget, sv^0, a^0, z^0, sz^0, tau^0, stau^0,
        %         v^2_lure, v^2_target, v^2_nontarget, sv^2, a^2, z^2, sz^2, tau^2, stau^2)
        
        others_blk0_idx = 4:9;
        others = (Block_j == 0)*z_j(others_blk0_idx) + (Block_j == 1)*z_j(9+others_blk0_idx);    

        Psi_j = [v_ij others];

end
