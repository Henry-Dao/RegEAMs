function Psi_j = Matching_Parameters_HDDM(MLE_model,subject_j,alpha_j_r)

% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com
%%     
% ---------- Transform random effects to the natural form ---------
    
     
    z_j = MLE_model.T_inv(alpha_j_r);
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
        
        others_blk0_idx = 4:9;
        others = (Block_j == 0)*z_j(others_blk0_idx) + (Block_j == 1)*z_j(9+others_blk0_idx);    

        Psi_j = [v_ij others];

end
