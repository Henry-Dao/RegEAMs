

load('VB_result.mat');

    lambda_best = VB_results.lambda;
    mu = lambda_best.mu; B = lambda_best.B;   d = lambda_best.d;   
    N = 1000;

    r = VB_settings.r;
    J = model.num_subjects;
    D_alpha = sum(model.subject_param_dim);

    p = D_alpha*(J + 1);
    Sigma_Psi = model.prior_par.Psi;
    df = model.prior_par.v + D_alpha + J + 1;

    
    for i = 1:N
    i
    for k=1:J
        epsilon = randn(D_alpha,1);
        z = randn(r,1);
        theta_temp = mu{k,1} + B{k,1}*z + d{k,1}.*epsilon;
        ALPHA{k,1}(:,i) = theta_temp;
    end
    
    end
    
VB_results.theta_VB.alpha = ALPHA;    

load('HCP_fulldata_combined.mat')
data = scale_data(data,1);

T_pred = 1000;
J = length(data);
post_pred_data = cell(T_pred,1);

for i = 1:T_pred
    i
    post_pred_data{i,1}.data = cell(J,1);
    for j = 1:J   
        disp(['---------- Simulating replicate ',num2str(i),', subject ', num2str(j),' ------------'])
        n_j = data{j,1}.num_trials;
        post_pred_data{i,1}.data{j,1}.num_trials = n_j;
        post_pred_data{i,1}.data{j,1}.X = data{j,1}.X;
        post_pred_data{i,1}.data{j,1}.Blk = data{j,1}.Blk; % Blk = Block type (1 = 2-back, 0 = 0-back)
        post_pred_data{i,1}.data{j,1}.E = data{j,1}.E;

        alpha_j_r =  VB_results.theta_VB.alpha{j,1}(:,i)';
        data_subject_j = data{j,1};
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

        E_j = data_subject_j.E; % (1 = lure, 2 = target, 3 = nontarget)
        
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
        
        v_ij_target = (v_ij_target_blk0_lure + v_ij_target_blk0_target + v_ij_target_blk0_nontarget +...
                  v_ij_target_blk2_lure + v_ij_target_blk2_target + v_ij_target_blk2_nontarget);

        v_ij_nontarget = (v_ij_nontarget_blk0_lure + v_ij_nontarget_blk0_target + v_ij_nontarget_blk0_nontarget +...
                  v_ij_nontarget_blk2_lure + v_ij_nontarget_blk2_target + v_ij_nontarget_blk2_nontarget);
        
        % ----------- Simulate data -----------------

        [post_pred_data{i,1}.data{j,1}.RE, post_pred_data{i,1}.data{j,1}.RT] = ...
            LBA_Simulation(A_ij,b_ij,[v_ij_nontarget v_ij_target],...
            tau_ij,sv_ij,true); % RE :  1 = nontarget, 2 = target
        outlier_idx = rand(n_j,1)<(1-0.98);
        post_pred_data{i,1}.data{j,1}.RE(outlier_idx) = randsample([1 2],sum(outlier_idx),true);
        post_pred_data{i,1}.data{j,1}.RT(outlier_idx) = 7*rand(sum(outlier_idx),1);
        post_pred_data{i,1}.data{j,1}.RE = post_pred_data{i,1}.data{j,1}.RE - 1; % RE :  0 = nontarget, 1 = target
    end  
end    
save('Post_Pred_data_VB.mat','post_pred_data')

