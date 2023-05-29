function Psi_j = Matching_Parameters_HLBA(MLE_model,data_subject_j,alpha_j_r)

% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com
%%     
% ---------- Transform random effects to the natural form ---------
   
    
    c_blk0 = exp(alpha_j_r(:,1));
    c_blk2 = exp(alpha_j_r(:,8));

    A_blk0 =  exp(alpha_j_r(:,2));  A_blk2 =  exp(alpha_j_r(:,2+7));
    tau_blk0 =  exp(alpha_j_r(:,7));  tau_blk2 =  exp(alpha_j_r(:,7+7));

% ---------- Match the parameters with block conditions ---------   
        Block_j = data_subject_j.Blk;
        b_ij = repmat((Block_j == 0)*(c_blk0 + A_blk0) + (Block_j == 1)*(c_blk2 + A_blk2),1,2);
        A_ij = repmat((Block_j == 0)*A_blk0 + (Block_j == 1)*A_blk2,1,2);
        tau_ij = repmat((Block_j == 0)*tau_blk0 + (Block_j == 1)*tau_blk2,1,2);
        sv_ij = ones(data_subject_j.num_trials,2);

% ---------- Match the parameters with block & exp conditions ---------   

% alpha_j_r = [log(c0), log(A0), log(v0), log(v0_lure), log(v0_target), log(v0_nontarget), log(tau0),
%              log(c2), log(A2), log(v2), log(v2_lure), log(v2_target), log(v2_nontarget), log(tau2)]
        E_j = data_subject_j.E;
        v_ij_target_blk0_lure = (exp(alpha_j_r(:,3)) - exp(alpha_j_r(:,4)))*(E_j == 1).*(Block_j == 0);
        v_ij_nontarget_blk0_lure = (exp(alpha_j_r(:,3)) + exp(alpha_j_r(:,4)))*(E_j == 1).*(Block_j == 0);
        
        v_ij_target_blk0_target = (exp(alpha_j_r(:,3)) + exp(alpha_j_r(:,5)))*(E_j == 2).*(Block_j == 0);
        v_ij_nontarget_blk0_target = (exp(alpha_j_r(:,3)) - exp(alpha_j_r(:,5)))*(E_j == 2).*(Block_j == 0);

        v_ij_target_blk0_nontarget = (exp(alpha_j_r(:,3)) - exp(alpha_j_r(:,6)))*(E_j == 3).*(Block_j == 0);
        v_ij_nontarget_blk0_nontarget = (exp(alpha_j_r(:,3)) + exp(alpha_j_r(:,6)))*(E_j == 3).*(Block_j == 0);

        v_ij_target_blk2_lure = (exp(alpha_j_r(:,10)) - exp(alpha_j_r(:,11)))*(E_j == 1).*(Block_j == 1);
        v_ij_nontarget_blk2_lure = (exp(alpha_j_r(:,10)) + exp(alpha_j_r(:,11)))*(E_j == 1).*(Block_j == 1);

        v_ij_target_blk2_target = (exp(alpha_j_r(:,10)) + exp(alpha_j_r(:,12)))*(E_j == 2).*(Block_j == 1);
        v_ij_nontarget_blk2_target = (exp(alpha_j_r(:,10)) - exp(alpha_j_r(:,12)))*(E_j == 2).*(Block_j == 1);

        v_ij_target_blk2_nontarget = (exp(alpha_j_r(:,10)) - exp(alpha_j_r(:,13)))*(E_j == 3).*(Block_j == 1);
        v_ij_nontarget_blk2_nontarget = (exp(alpha_j_r(:,10)) + exp(alpha_j_r(:,13)))*(E_j == 3).*(Block_j == 1);
        
        %------- match the drift rates with the observed responses --------
        
        v_ij_c = (v_ij_target_blk0_lure + v_ij_target_blk0_target + v_ij_target_blk0_nontarget +...
                  v_ij_target_blk2_lure + v_ij_target_blk2_target + v_ij_target_blk2_nontarget).*(data_subject_j.RE == 1) +...
                  (v_ij_nontarget_blk0_lure + v_ij_nontarget_blk0_target + v_ij_nontarget_blk0_nontarget +...
                  v_ij_nontarget_blk2_lure + v_ij_nontarget_blk2_target + v_ij_nontarget_blk2_nontarget).*(data_subject_j.RE == 0);

        v_ij_k = (v_ij_target_blk0_lure + v_ij_target_blk0_target + v_ij_target_blk0_nontarget +...
                  v_ij_target_blk2_lure + v_ij_target_blk2_target + v_ij_target_blk2_nontarget).*(data_subject_j.RE == 0) +...
                  (v_ij_nontarget_blk0_lure + v_ij_nontarget_blk0_target + v_ij_nontarget_blk0_nontarget +...
                  v_ij_nontarget_blk2_lure + v_ij_nontarget_blk2_target + v_ij_nontarget_blk2_nontarget).*(data_subject_j.RE == 1);

        v_ij = [v_ij_c v_ij_k];
        Psi_j = [b_ij, A_ij,v_ij,sv_ij,tau_ij];
       
end