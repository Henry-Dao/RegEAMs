function alpha = CMC_alpha(model,data,theta_G,alpha,proposals_alpha,MCMC_setting)

J = length(data);
D_alpha = model.subject_param_dim;
beta_vec = theta_G.beta_vec;
mu = theta_G.mu;
[chol_covmat,cond_num] = chol(theta_G.Sigma,'lower');
Sigma = theta_G.Sigma;
if cond_num ~=0
    disp('CMC alpha: Sigma_alpha is not positively definite !');
    Sigma = topdm(theta_G.Sigma);
    chol_covmat = chol(Sigma,'lower');
    Sigma = chol_covmat*chol_covmat';
end
C_star = chol_covmat;     C_star(1:D_alpha+1:end) = log(diag(chol_covmat));
vech_C_star = vech(C_star);

w_mix1 = MCMC_setting.CMC_alpha_w1; % w_mix1 = weight for estimated proposal
w_mix2 = MCMC_setting.CMC_alpha_w2; % w_mix2 = weight for prior
w_mix3 = 1 - w_mix1 - w_mix2; % w_mix3 = weight for third component
w_mix = MCMC_setting.CMC_alpha_w_mix; % setting the weights of the mixture in the burn in and initial sampling stage.

R = MCMC_setting.CMC_alpha_number_particles;
epsilon = MCMC_setting.CMC_alpha_epsilon;

display_warnings = MCMC_setting.display_warnings;
display_particles = MCMC_setting.display_particles;
filtering_alpha = MCMC_setting.filtering_alpha;
data_stacking = MCMC_setting.stacking;
parfor j=1:J
    n_j = data{j,1}.num_trials;

    % ------- (step 1): Generate alpha_j from the proposal distribution -------
    %
    % -------------------------------------------------------------------------
    
    alpha_j_k = alpha(:,j); % the set of random effects from previous iteration of MCMC for conditioning.
    
    if proposals_alpha{j,1}.better_proposal == true % use a better proposal
        % setting the weight of the mixture for the proposal in the sampling stage.
        
        % generating the proposals from the mixture distribution in the sampling stage
        %-----------------------
        u = rand(R-1,1);
        n1 = sum(u < w_mix1);
        n2 = sum((w_mix1 < u) & (u < w_mix1 + w_mix2));
        n3 = R-1-n1-n2;
        %       Denote:  x1 = alpha_j and x2 = theta_G = (mu vech_C*);
        x2 = [mu; vech_C_star; beta_vec];
        % mean_prop(:,j) = [alpha_j; mu; vech(C^*)]
        mu_1 = proposals_alpha{j,1}.mean(1:D_alpha); % mean of x1;
        mu_2 = proposals_alpha{j,1}.mean(D_alpha+1:end); % mean of x2
        
        S_11 = proposals_alpha{j,1}.cov(1:D_alpha,1:D_alpha);  % cov(x1,x1)
        S_22 = proposals_alpha{j,1}.cov(D_alpha+1:end, D_alpha+1:end);  % cov(x2,x2)
        S_12 = proposals_alpha{j,1}.cov(1:D_alpha,D_alpha+1:end); % cov(x1,x2)
        S_21 = S_12';
        M =  S_12/S_22;
        
        cond_mean = mu_1 + M*(x2-mu_2); % cond_mean = mean of x1|x2, this is the mean of the proposal in the sampling stage
        cond_var = S_11 - M*S_21; % computing the variance of the proposal in the sampling stage
        [chol_cond_var,cond_num] = chol(cond_var,'lower');
        if cond_num ~=0
            if display_warnings == true
                disp(['CMC alpha: subject ', num2str(j),' has conditional covariance matrix is not positively definite !']);
            end
            cond_var = topdm(cond_var);
            chol_cond_var = chol(cond_var,'lower');
            cond_var = chol_cond_var*chol_cond_var';
        end

        rnorm1 = cond_mean + chol_cond_var*randn(D_alpha,n1);
        rnorm3 = alpha_j_k + chol_cond_var*randn(D_alpha,n3);     
        
        rnorm2 = mu + chol_covmat*randn(D_alpha,n2);
        
        alpha_j_R1 = [rnorm1 rnorm2 rnorm3]; % alpha_j_R = [alpha_j^1, ... alpha_j^R]. Particles alpha_j^r are stores in colums of matrix alpha_j_R
        %-----------------------
    else
        % generating the proposals from the mixture distribution in the burn in
        % and initial sampling stage
        %-----------------------
        
        u = rand(R-1,1);
        id1 = (u<w_mix);
        n1 = sum(id1);
        n2 = R-1-n1;
%         chol_covmat = chol(Sigma,'lower'); % brought outside
%         parfor
        rnorm1 = alpha_j_k + sqrt(epsilon).*chol_covmat*randn(D_alpha,n1);
        rnorm2 = mu + chol_covmat*randn(D_alpha,n2);
        
        alpha_j_R1 = [rnorm1 rnorm2];  % alpha_j_R = [alpha_j^1, ... alpha_j^R]. Particles alpha_j^r are stores in colums of matrix alpha_j_R
        
        %------------------------
    end
    
    % set the first particles to the values of the random effects from the
    % previous iterations of MCMC for conditioning
    alpha_j_R = [alpha_j_k alpha_j_R1]'; % particles are stored in rows !
    if filtering_alpha == true
        alpha_j_R = filter_alpha(alpha_j_R,-10,10); % throw away bad particles
    end
    % -------------- (step 2): Compute the importance weights  ----------------
    %
    % -------------------------------------------------------------------------
    
    % Duplicate data (y_j) and stack them into a column vector
    if data_stacking == true
        Y_j_stacked = model.Stacking_obs(data{j,1},size(alpha_j_R,1)); % model_stacked
        u_ij_R = model.structural_equation(model,data{j,1},alpha_j_R,beta_vec);
        z_ij_R = model.matching_function_1(model,Y_j_stacked,u_ij_R);
        
        % Compute the log p(y_j|alpha_j^r,theta_G)
        
        pdf_j = model.density(model,Y_j_stacked,z_ij_R,false);
        
        lw_reshape = reshape(pdf_j.logs,n_j,size(alpha_j_R,1));
        logw_first = sum(lw_reshape)'; %
    else
        logw_first = zeros(size(alpha_j_R,1),1);
        for r = 1:size(alpha_j_R,1)
            z_ij_r = model.matching_parameters(model,data{j,1},alpha_j_R(r,:),beta_vec);
            pdf_j = model.density(model,data{j,1},z_ij_r,false);
            logw_first(r,1) = pdf_j.log;
        end

    end
    % Computing the log of p(\alpha|\theta) and density of the proposal for
    %burn in and initial sampling stage (count<=switch_num) and sampling
    %stage (count>switch_num)
    
    if  proposals_alpha{j,1}.better_proposal == true
        
        cond_var = (cond_var + cond_var')/2; % to avoid this error: 

        logw_second = log(mvnpdf(alpha_j_R,mu',Sigma));
        logw_third = log(w_mix1.*mvnpdf(alpha_j_R,cond_mean',cond_var)+...
            w_mix2.*mvnpdf(alpha_j_R,mu',Sigma)+ w_mix3.*mvnpdf(alpha_j_R,alpha_j_k',cond_var));
        logw = logw_first + logw_second - logw_third;
    else
        %         logw_second = logmvnpdf(alpha_j_R,mu,Sigma);
        logw_second = log(mvnpdf(alpha_j_R,mu',Sigma));
        logw_third = log(w_mix.*mvnpdf(alpha_j_R,alpha_j_k',epsilon*Sigma)+...
            (1-w_mix).*mvnpdf(alpha_j_R,mu',Sigma));
        logw = logw_first + logw_second - logw_third;
    end
    
    idx1 = ~isinf(logw);
    idx2 = ~isnan(logw);
    idx3 = imag(logw)==0;
    idx = idx1&idx2&idx3;
    if sum(idx) == 0
        if display_particles == true
            disp(['     CMC alpha: Subject ',num2str(j),' all -log-likelihood are -Inf/NaN/Complex numbers']);
        end
        alpha(:,j) = alpha_j_k;
    else
        logw = logw(idx);
        
        alpha_j_R = alpha_j_R(idx,:);
        max_logw = max(logw);
        weight = exp(logw-max_logw);
        weight = weight./sum(weight);
        
        Nw = length(weight);
        ind = randsample(Nw,1,true,weight);
        alpha(:,j) = alpha_j_R(ind,:);
        if display_particles == true
            disp(['     CMC alpha: Subject ',num2str(j),' has ',num2str(sum(idx)),' finite log-likelihoods and the selected particle is particle ',num2str(ind)]);
        end
    end
    %----------------------------------------------------------------------------------------------------------------------------------
    
end

end

