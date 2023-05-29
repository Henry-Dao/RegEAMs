
%% Description: 
    J = length(data); 	
    model.num_subjects = J;     
    
    model.structure = "Regression"; % "Regression" or "Nonhierarchical" or "Hierarchical"
    model.prior_Sigma_alpha = "Informative"; % "Informative" or "Marginally noninformative"
    model.subject_param_dim = 14;% number of random effects per participant (D_alpha)

% ----------------------------------------------------------
      
    D_alpha = model.subject_param_dim;
    model.beta_dim = [14,length(data{1,1}.X)];
    model.prior_par.mu_beta = zeros(prod(model.beta_dim),1);
    model.prior_par.Sigma_beta = 10*eye(prod(model.beta_dim));

    model.prior_par.mu = zeros(D_alpha,1); 
    model.prior_par.cov = eye(D_alpha); 
    
    % hyperparameters for a noninformative prior for Sigma:
    model.prior_par.v_a = 2; 
    model.prior_par.A_d = ones(D_alpha,1);
    
    % hyperparameters for an informative prior for Sigma:
    model.prior_par.v = D_alpha+1;    
    model.prior_par.Psi = eye(D_alpha);
    
    
    model.density = str2func("LBA_pdf"); 
    model.mixture_weight = 0.98;
    model.p_0 = 1/(2*7);
    model.matching_parameters = str2func("Matching_Parameters_RegLBA"); 
    model.matching_gradients = str2func("Matching_Gradients_RegLBA"); 

%% Initialization for VB and MCMC:
% alpha = [log(c0), log(A0), log(v0), log(v0_lure), log(v0_target), log(v0_nontarget), log(tau0),
%          log(c2), log(A2), log(v2), log(v2_lure), log(v2_target), log(v2_nontarget), log(tau2)]

    minRT = min(data{1,1}.RT);
    for j = 2:J
        if minRT > min(data{j,1}.RT)
            minRT = min(data{j,1}.RT);
        end
    end
    
    v0_lure = 1.09;   v0_target = 1.5;  v0_nontarget = 2;
    max_v0s = max([v0_lure, v0_nontarget, v0_target]);
    v0 = rand() + max_v0s;
    c0 = 1.5;
    A0 = 0.1;   
    tau0 = 0.36;

    v2_lure = 1.09;   v2_target = 1.4;  v2_nontarget = 1.9;
    max_v2s = max([v2_lure, v2_nontarget, v2_target]);
    v2 = rand() + max_v2s;
    c2 = 0.8;
    A2 = 0.1;   
    tau2 = 0.36;

    initial_mu_alpha = [log(c0), log(A0), log(v0), log(v0_lure), log(v0_target), log(v0_nontarget), log(tau0),...
         log(c2), log(A2), log(v2), log(v2_lure), log(v2_target), log(v2_nontarget), log(tau2)];

    initial_Sigma_alpha = iwishrnd(model.prior_par.Psi,model.prior_par.v);
    %initial_alpha = mvnrnd(initial_mu_alpha',initial_Sigma_alpha,J)';
    initial_alpha = initial_mu_alpha'*ones(1,J);

    initial_log_a_d = zeros(D_alpha,1);  
    initial_beta_vec = zeros(prod(model.beta_dim),1);
    if model.prior_Sigma_alpha == "Informative"
        VB_settings.Domain_Knowledge_initial = [initial_alpha(:); initial_beta_vec; initial_mu_alpha(:)];
    elseif model.prior_Sigma_alpha == "Marginally noninformative"
        VB_settings.Domain_Knowledge_initial = [initial_alpha(:); initial_beta_vec; initial_mu_alpha(:); initial_log_a_d];
    end

    MCMC_setting.initialization.beta_vec = initial_beta_vec;
    MCMC_setting.initialization.alpha = initial_alpha;
    MCMC_setting.initialization.mu = initial_mu_alpha; %the initial values for parameter \mu
    MCMC_setting.initialization.Sigma = initial_Sigma_alpha; % the initial values for \Sigma
    MCMC_setting.initialization.a_d = 1./random('gam',1/2,1,D_alpha,1);

