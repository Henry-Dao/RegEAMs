function MCMC_output = PMwG(model,data,MCMC_setting)   
%% PMWG_GENERAL Particle Metropolis within Gibbs sampler
%% 
% * Author: Viet-Hung Dao
% * Date: 30 Oct 2022
% * viethung.unsw@gmail.com
% Ref: https://arxiv.org/abs/2302.10389
%% PMwG - Description

            % Set the MCMC tuning parameters
    N_burn = MCMC_setting.burnin;     % the burn in iterations
    N_adapt = MCMC_setting.adaptation_stage; % number of draws needed to estimate the covariance matrix in the proposal
    N_sampling = MCMC_setting.start_sampling;    % we take the last 10000 draws
    N_threshold = MCMC_setting.stop_update; % after N_threshold, stop updating the proposal 
    N_iter = N_threshold + N_sampling;     % the maximum total number of iterations
    T_update = MCMC_setting.update_period;  % update the proposal after 20 iterations      
% Allocation Memory
    D_alpha = model.subject_param_dim;% number of random effects per participant (D_alpha); % number of random effects per participant    
    J = length(data); % number of participants  
    alpha_store = zeros(D_alpha,J,N_iter); % 3-D matrix that stores random effects draws
    proposals_alpha = MCMC_setting.proposals_alpha;
% for compatibility with later functions (ex: save() ) we still need to define the following variables    
    mu_store = [];  vech_Sigma_alpha = []; vech_C_star_store = [];
    a_d_store = []; 
    beta_vec_store = [];  
    if model.structure == "Regression"
        d = model.beta_dim(2); % number of columns of beta (number of covariates)
        d_alpha = model.beta_dim(1); % number of rows of beta
        beta_vec_store = zeros(d_alpha*d,N_iter);
        proposals_beta = MCMC_setting.proposals_beta;
        X = data{1,1}.X;
        for j = 2:J
            X = [X; data{j,1}.X];
        end
        temp = inv(X'*X);
        
        Sigma_beta0 = kron(eye(d_alpha),temp); % d_alpha = number of linked variables
        MCMC_setting.Sigma_beta0 = Sigma_beta0;
    end
    if model.structure ~= "Nonhierarchical"        
        mu_store = zeros(D_alpha,N_iter);  
        vech_C_star_store = zeros(D_alpha*(D_alpha+1)/2,N_iter); % C^* is tranformed !
        vech_Sigma_alpha = zeros(D_alpha*(D_alpha+1)/2,N_iter); %
        if model.prior_Sigma_alpha == "Marginally noninformative"      
            a_d_store = zeros(D_alpha,N_iter);
        end   
    end
% Initialization

    count = 1;
    theta_G = []; % for compatibility in CMC_alpha and CMC_beta;
    if model.structure == "Regression"
        theta_G.beta_vec = MCMC_setting.initialization.beta_vec;
    end
    if model.structure ~= "Nonhierarchical"
        theta_G.mu = MCMC_setting.initialization.mu; %the initial values for parameter \mu
        theta_G.Sigma = MCMC_setting.initialization.Sigma; % the initial values for \Sigma
        theta_sig2_inv = inv(theta_G.Sigma);
        chol_sig2 = chol(theta_G.Sigma,'lower');
        theta_G.a_d = MCMC_setting.initialization.a_d;
    end
    alpha = MCMC_setting.initialization.alpha;
% Stack data
    if MCMC_setting.stacking == true
        data.data_stacked = model.stack_data(data);
    end
%% PMwG - Particles Metropolis within Gibbs
    t = 1;
    tic
    xx = toc;
    while t<=N_iter
        if model.structure ~= "Nonhierarchical"
   
        % ------------------- Sample \mu|rest in Gibbs step --------------------
            var_mu = (J*(theta_G.Sigma\eye(D_alpha)) + (model.prior_par.cov)\eye(D_alpha))\eye(D_alpha); 
            mean_mu = var_mu*( theta_G.Sigma\sum(alpha,2) + model.prior_par.cov\model.prior_par.mu);
           
            [L,temp] = chol(var_mu,'lower');
            if temp == 0  
                theta_G.mu = mean_mu + L*randn(D_alpha,1);
            end
        
        % ------------------ Sample \Sigma|rest in Gibbs step --------------------
            
            cov_temp = zeros(D_alpha);
            for j=1:J
                cov_temp = cov_temp + (alpha(:,j)-theta_G.mu)*(alpha(:,j)-theta_G.mu)';
            end
            if model.prior_Sigma_alpha == "Marginally noninformative"      
                k_a = model.prior_par.v_a + D_alpha - 1 + J;
                B_a = 2*(model.prior_par.v_a)*diag(1./theta_G.a_d) + cov_temp;
            elseif model.prior_Sigma_alpha == "Informative"
                k_a = model.prior_par.v + J;
                B_a = model.prior_par.Psi + cov_temp;
            end
            
        %     theta_G.Sigma = iwishrnd(B_a,k_a);  
            sig2_temp = iwishrnd(B_a,k_a); 
            [chol_sig2_temp,p_test] = chol(sig2_temp,'lower');
            if p_test == 0
                theta_G.Sigma = sig2_temp;    
                chol_sig2 = chol_sig2_temp;
                theta_sig2_inv = inv(theta_G.Sigma);
            else
                if MCMC_setting.display_warnings == true
                    disp(' Sigma_alpha is note positive definite ')
                end
            end
        % -------------- Sample a_{1},...,a_{D_alpha}|rest in Gibbs step ----------------
            if model.prior_Sigma_alpha == "Marginally noninformative"      
                theta_G.a_d = 1./gamrnd((model.prior_par.v_a + D_alpha)/2, 1./(model.prior_par.v_a*diag(theta_sig2_inv) + (1./model.prior_par.A_d).^2 ));
            end      
        end
% Update the proposals: $m_j(\alpha_j|y_j,\mu_{\alpha_j}^{(t)},\Sigma_{\alpha_j}^{(t)})$ and $m_j(\beta_{\text{vec}}|y_{1:J},\alpha_{1:J}^{(t)})$
    % ----------------- Sample alpha_j|rest in Gibbs step ---------------------- 
       if (t > N_burn + N_adapt + (count-1)*T_update) && (t<= N_threshold)% After first updating, the proposal is reestimated after very 20 iterations
             if (t - N_burn < MCMC_setting.number_samples)
                t_start = N_burn + 1;
             else
                t_start = t - MCMC_setting.number_samples;
             end
             t_end = t-1; 
             
             for j=1:J 
                 if model.structure == "Nonhierarchical"
                     theta_j = reshape(alpha_store(:,j,t_start:t_end),D_alpha,t_end-t_start+1);
                 elseif model.structure == "Hierarchical"
                     theta_j = [reshape(alpha_store(:,j,t_start:t_end),D_alpha,t_end-t_start+1); mu_store(:,t_start:t_end); vech_C_star_store(:,t_start:t_end)];
                 else
                     theta_j = [reshape(alpha_store(:,j,t_start:t_end),D_alpha,t_end-t_start+1); mu_store(:,t_start:t_end); vech_C_star_store(:,t_start:t_end); beta_vec_store(:,t_start:t_end)];
                 end
                 proposals_alpha{j,1}.cov = cov(theta_j'); %computing sample covariance matrix for the joint random effects and parameters \mu_{\alpha} and \Sigma_{\alpha}
                 [~,cond_num] = chol(proposals_alpha{j,1}.cov);
                 if cond_num ~=0
                     if MCMC_setting.display_warnings == true
                        disp(['Proposal distribution for alpha: subject ',num2str(j), ' has non positively definite covariance'])
                     end
                     proposals_alpha{j,1}.cov = topdm(proposals_alpha{j,1}.cov); %little correction if the covariance matrix for the proposal is not positive definite matrix.
                 end 
                 proposals_alpha{j,1}.mean = mean(theta_j,2); %computing the sample mean for the joint random effects and parameters \mu_{\alpha} and \Sigma_{\alpha}, NO log(a)
                 proposals_alpha{j,1}.better_proposal = true;
             end
             
             if model.structure == "Regression"
                 theta = beta_vec_store(:,t_start:t_end);
                 for j = 1:J
                     theta = [theta; reshape(alpha_store(:,j,t_start:t_end),D_alpha,t_end-t_start+1)];
                 end
                 proposals_beta.cov = cov(theta'); % Sigma_beta_hat
                 [~,cond_num] = chol(proposals_beta.cov);
                 if cond_num ~=0
                     if MCMC_setting.display_warnings == true
                        disp('Proposal distribution for beta: has non positive definite covariance')
                     end
                     proposals_beta.cov = topdm(proposals_beta.cov); % little correction if the covariance matrix for the proposal is not positive definite matrix.
                 end 
  
                 proposals_beta.mean = mean(theta,2); % mu_beta_hat
                 proposals_beta.better_proposal = true;
             end
           % reduce the number of particles   
          count = count+1;        
       end
       if MCMC_setting.display_iter == true
           disp(['iteration ',num2str(t),'|| proposal ',num2str(count)]);
       end
       if t>MCMC_setting.when_to_reduce_the_numer_particles
           MCMC_setting.CMC_alpha_number_particles = MCMC_setting.CMC_alpha_number_particles_after;
           MCMC_setting.CMC_beta_number_particles = MCMC_setting.CMC_beta_number_particles_after;
       end
% CMC Sample $\alpha_j \sim m_j(\alpha_j|y_j,\mu_{\alpha_j}^{(t)},\Sigma_{\alpha_j}^{(t)})$ 
       [alpha] = MCMC_setting.CMC_alpha(model,data,theta_G,alpha,proposals_alpha,MCMC_setting);
% CMC Sample $\beta_{\text{vec}} \sim m_j(\beta_{\text{vec}}|y_{1:J},\alpha_{1:J}^{(t)})$ 
       if model.structure == "Regression" 
           [theta_G.beta_vec] = MCMC_setting.CMC_beta(model,data,theta_G,alpha,proposals_beta,MCMC_setting);
       end                                          
% Store MCMC draws:
    %   --------------------- storing the MCMC draws  -------------------------    
        % storing random effects    
        alpha_store(:,:,t) = alpha;
        if model.structure ~="Nonhierarchical"
            % storing the global parameters
            mu_store(:,t) = theta_G.mu;
    
            C_star = chol_sig2;     C_star(1:D_alpha+1:end) = log(diag(chol_sig2));
    
            vech_C_star_store(:,t) = vech(C_star);
            vech_Sigma_alpha(:,t) = vech(theta_G.Sigma);
            proposals.alpha = proposals_alpha;
        end
        if model.structure =="Regression"
            beta_vec_store(:,t) = theta_G.beta_vec;
            proposals.beta = proposals_beta;
        end
        if model.prior_Sigma_alpha =="Marginally noninformative"
            a_d_store(:,t) = theta_G.a_d;
        end
        % save the output to your directory
        if MCMC_setting.save == true
            if mod(t,MCMC_setting.save_period)==0
                save([MCMC_setting.save_path MCMC_setting.save_name '.mat'],'beta_vec_store','mu_store','vech_Sigma_alpha','vech_C_star_store','a_d_store','alpha_store','proposals','t');
            end
        end
        if MCMC_setting.display_iter_running_time == true           
            disp(['-------------- Running time is ',num2str(round(toc-xx,1)), ' seconds --------------']);
            xx = toc;
        end
        t=t+1;   
    end
    MCMC_output.alpha = alpha_store;
    MCMC_output.beta_vec = beta_vec_store;
    MCMC_output.mu_alpha = mu_store;
    MCMC_output.vech_Sigma_alpha = vech_Sigma_alpha;
    MCMC_output.vech_C_alpha_star = vech_C_star_store;
    MCMC_output.a_d = a_d_store;
    
    MCMC_output.CPU_time = toc;
end