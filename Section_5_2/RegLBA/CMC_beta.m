function beta_vec = CMC_beta(model,data,theta_G,alpha,proposals_beta,MCMC_setting)

% ------- (step 1): Generate Beta from the proposal distribution -------
%
% -------------------------------------------------------------------------
J = length(data);
d = model.beta_dim(2);
d_alpha = model.beta_dim(1);

Sigma_beta0 = MCMC_setting.Sigma_beta0;
beta_vec_k = theta_G.beta_vec; % beta from previous iteration of MCMC for conditioning.

w_mix1 = MCMC_setting.CMC_beta_w1; % w_mix1 = weight for estimated proposal
w_mix2 = MCMC_setting.CMC_beta_w2; % w_mix2 = weight for prior
w_mix3 = 1 - w_mix1 - w_mix2; % w_mix3 = weight for third component
w_mix = MCMC_setting.CMC_beta_w_mix; % setting the weights of the mixture in the burn in and initial sampling stage.

R = MCMC_setting.CMC_beta_number_particles;
epsilon = MCMC_setting.CMC_beta_epsilon;
if proposals_beta.better_proposal == true % use a better proposal
    w_mix3 = 1 - w_mix1 - w_mix2; % w_mix3 = weight for third component
    % generating the proposals from the mixture distribution in the sampling stage
    %-----------------------
    u = rand(R-1,1);
    n1 = sum(u < w_mix1);
    n2 = sum((w_mix1 < u) & (u < w_mix1 + w_mix2));
    n3 = R-n1-n2-1;
    
    %       Denote:  x1 = beta and x2 = alpha = (alpha_1,..., alpha_J);
    x2 = alpha(:);
    % mean_prop(:,j) = [alpha_j; mu; vech(C^*)]
    mu_1 = proposals_beta.mean(1:d_alpha*d); % mean of x1;
    mu_2 = proposals_beta.mean(d_alpha*d+1:end); % mean of x2
    
    S_11 = proposals_beta.cov(1:d_alpha*d,1:d_alpha*d);  % cov(x1,x1)
    S_22 = proposals_beta.cov(d_alpha*d+1:end, d_alpha*d+1:end);  % cov(x2,x2)
    S_12 = proposals_beta.cov(1:d_alpha*d,d_alpha*d+1:end); % cov(x1,x2)
    S_21 = S_12';
    M =  S_12/S_22;
    
    cond_mean = mu_1 + M*(x2-mu_2); % cond_mean = mean of x1|x2, this is the mean of the proposal in the sampling stage
    cond_var = S_11 - M*S_21; % computing the variance of the proposal in the sampling stage
    
    [chol_cond_var,cond_num] = chol(cond_var,'lower');
    if cond_num ~=0
        if MCMC_setting.display_warnings == true
            disp('CMC beta: conditional covariance matrix is not positively definite !');
        end
        cond_var = topdm(cond_var);      
        chol_cond_var = chol(cond_var,'lower');
    end

    rnorm1 = cond_mean + chol_cond_var*randn(d_alpha*d,n1);
    rnorm3 = beta_vec_k + chol_cond_var*randn(d_alpha*d,n3);
    

    rnorm2 = mvnrnd(model.prior_par.mu_beta',model.prior_par.Sigma_beta,n2);
    
    beta_vec_R1 = [rnorm1'; rnorm2; rnorm3']; % Particles beta_vec^(r) are stores in rows of matrix beta_vec_R
    %-----------------------
else
    % generating the proposals from the mixture distribution in the burn in
    % and initial sampling stage

    %-----------------------

    u = rand(R-1,1);
    id1 = (u<w_mix);
    n1 = sum(id1);
    n2 = R-n1-1;
    

    rnorm1 = mvnrnd(beta_vec_k',epsilon*Sigma_beta0,n1);
    rnorm2 = mvnrnd(model.prior_par.mu_beta',model.prior_par.Sigma_beta,n2);
    
    beta_vec_R1 = [rnorm1; rnorm2];  % Particles beta_vec^(r) are stores in rows of matrix beta_vec_R
    
    %------------------------
end

% set the first particles to the values of the random effects from the
% previous iterations of MCMC for conditioning
beta_vec_R = [beta_vec_k'; beta_vec_R1]; %particles are stored in rows

% -------------- (step 2): Compute the importance weights  ----------------
%
% -------------------------------------------------------------------------
logw_first = zeros(R,1);

parfor r = 1:R
    for j = 1:J
        z_ij_r = model.matching_parameters(model,data{j,1},alpha(:,j)',beta_vec_R(r,:));       
        pdf_j = model.density(model,data{j,1},z_ij_r,false);
        logw_first(r) = logw_first(r) + pdf_j.log;
    end
end

% Computing the log of p(\alpha|\theta) and density of the proposal for
%burn in and initial sampling stage (count<=switch_num) and sampling
%stage (count>switch_num)

if  proposals_beta.better_proposal == true
            
    cond_var = (cond_var + cond_var')/2; % to avoid this error: 

%     logw_second = logmvnpdf(beta_vec_R,theta_G.mu,theta_G.sig2);
    logw_second = log(mvnpdf(beta_vec_R,model.prior_par.mu_beta',model.prior_par.Sigma_beta));
    logw_third = log(w_mix1.*mvnpdf(beta_vec_R,cond_mean',cond_var)+...
        w_mix2.*mvnpdf(beta_vec_R,model.prior_par.mu_beta',model.prior_par.Sigma_beta)+...
        w_mix3.*mvnpdf(beta_vec_R,beta_vec_k',cond_var));
    logw = logw_first + logw_second - logw_third;
else
    %         logw_second = logmvnpdf(beta_vec_R,theta_G.mu,theta_G.sig2);
    logw_second = log(mvnpdf(beta_vec_R,model.prior_par.mu_beta',model.prior_par.Sigma_beta));
    logw_third = log(w_mix.*mvnpdf(beta_vec_R,beta_vec_k',epsilon.*Sigma_beta0)+...
        (1-w_mix).*mvnpdf(beta_vec_R,model.prior_par.mu_beta',model.prior_par.Sigma_beta));
    logw = logw_first + logw_second - logw_third;
end

idx1 = ~isinf(logw);
idx2 = ~isnan(logw);
idx3 = imag(logw)==0;
idx = idx1&idx2&idx3;
if sum(idx) == 0
    if MCMC_setting.display_particles == true
        disp('     CMC beta: all -log-likelihood are -Inf/NaN/Complex numbers');
    end
    beta_vec = beta_vec_k;
else
    logw = logw(idx);
    
    beta_vec_R = beta_vec_R(idx,:);
    max_logw = max(logw);
    weight = exp(logw-max_logw);
    weight = weight./sum(weight);
    
    Nw = length(weight);
    ind = randsample(Nw,1,true,weight);
    beta_vec = beta_vec_R(ind,:)';
    if MCMC_setting.display_particles == true
        disp(['     CMC beta: ',num2str(sum(idx)),' finite log-likelihoods and the selected particle is particle ',num2str(ind)]);
    end   
end
%----------------------------------------------------------------------------------------------------------------------------------
end

