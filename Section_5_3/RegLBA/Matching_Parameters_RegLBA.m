function Psi_j = Matching_Parameters_HLBA_HCPdata_model2(model,data_subject_j,alpha_j_r,beta_vec)

%% Input: alpha_j_r is a particle (row vector), not a matrix
% alpha_j_r = [log(c0), log(A0), log(v0), log(v0_lure), log(v0_target), log(v0_nontarget), log(tau0),
%              log(c2), log(A2), log(v2), log(v2_lure), log(v2_target), log(v2_nontarget), log(tau2)]
% beta_matrix = [log(c0); v_blk0_lure_target; v_blk0_lure_nontarget; v_blk0_target_target; v_blk0_target_nontarget; v_blk0_nontarget_target; v_blk0_nontarget_nontarget;
%                log(c2); v_blk2_lure_target; v_blk2_lure_nontarget; v_blk2_target_target; v_blk2_target_nontarget; v_blk2_nontarget_target; v_blk2_nontarget_nontarget] 
%% linking equation: 
%             log(c_ij) = log(c_blk) + beta_c*X_j
%             v_target_ij = v_blk +/- v_blk_cond + beta_blk_cond_target*X_j
%             v_nontarget_ij = v_blk +/- v_blk_cond + beta_blk_cond_nontarget*X_j
%% Output: a matrix, each row corresponds to a set of parameters (in natural scale) that goes into the pdf
% Psi_j = [b_ij, A_ij,v_ij,sv_ij,tau_ij] a matrix, each row
% corresponding to a set of 10 parameters that go into the LBA pdf

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

% ---------- Match the parameters with block conditions ---------   
        Block_j = data_subject_j.Blk;
        b_ij = repmat((Block_j == 0)*(c_blk0 + A_blk0) + (Block_j == 1)*(c_blk2 + A_blk2),1,2);
        A_ij = repmat((Block_j == 0)*A_blk0 + (Block_j == 1)*A_blk2,1,2);
        tau_ij = repmat((Block_j == 0)*tau_blk0 + (Block_j == 1)*tau_blk2,1,2);
        sv_ij = ones(data_subject_j.num_trials,2);

% ---------- Match the parameters with block & exp conditions ---------   

        E_j = data_subject_j.E;
        v_ij_target_blk0_lure = (exp(alpha_j_r(:,3)) - exp(alpha_j_r(:,4)) + betaX_j(2,:)')*(E_j == 1).*(Block_j == 0);
        v_ij_nontarget_blk0_lure = (exp(alpha_j_r(:,3)) + exp(alpha_j_r(:,4)) + betaX_j(3,:)')*(E_j == 1).*(Block_j == 0);
        
        v_ij_target_blk0_target = (exp(alpha_j_r(:,3)) + exp(alpha_j_r(:,5)) + betaX_j(4,:)')*(E_j == 2).*(Block_j == 0);
        v_ij_nontarget_blk0_target = (exp(alpha_j_r(:,3)) - exp(alpha_j_r(:,5)) + betaX_j(5,:)')*(E_j == 2).*(Block_j == 0);

        v_ij_target_blk0_nontarget = (exp(alpha_j_r(:,3)) - exp(alpha_j_r(:,6)) + betaX_j(6,:)')*(E_j == 3).*(Block_j == 0);
        v_ij_nontarget_blk0_nontarget = (exp(alpha_j_r(:,3)) + exp(alpha_j_r(:,6)) + betaX_j(7,:)')*(E_j == 3).*(Block_j == 0);

        v_ij_target_blk2_lure = (exp(alpha_j_r(:,10)) - exp(alpha_j_r(:,11)) + betaX_j(9,:)')*(E_j == 1).*(Block_j == 1);
        v_ij_nontarget_blk2_lure = (exp(alpha_j_r(:,10)) + exp(alpha_j_r(:,11)) + betaX_j(10,:)')*(E_j == 1).*(Block_j == 1);

        v_ij_target_blk2_target = (exp(alpha_j_r(:,10)) + exp(alpha_j_r(:,12)) + betaX_j(11,:)')*(E_j == 2).*(Block_j == 1);
        v_ij_nontarget_blk2_target = (exp(alpha_j_r(:,10)) - exp(alpha_j_r(:,12)) + betaX_j(12,:)')*(E_j == 2).*(Block_j == 1);

        v_ij_target_blk2_nontarget = (exp(alpha_j_r(:,10)) - exp(alpha_j_r(:,13)) + betaX_j(13,:)')*(E_j == 3).*(Block_j == 1);
        v_ij_nontarget_blk2_nontarget = (exp(alpha_j_r(:,10)) + exp(alpha_j_r(:,13)) + betaX_j(14,:)')*(E_j == 3).*(Block_j == 1);
        
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