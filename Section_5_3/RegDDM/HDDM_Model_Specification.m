
%% Description: 
    J = length(data); 	
    model.num_subjects = J;     
    for j = 1:J
        data{j,1}.RE = categorical(data{j,1}.RE,[0 1],{'lower' 'upper'});
    end

    
    model.structure = "Regression"; % "Regression" or "Nonhierarchical" or "Hierarchical"
    model.prior_Sigma_alpha = "Informative"; % "Informative" or "Marginally noninformative"
    model.subject_param_dim = 18;% number of random effects per participant (D_alpha)

% ----------------------------------------------------------
      
    D_alpha = model.subject_param_dim;
    model.beta_dim = [8,length(data{1,1}.X)];
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
    
    model.density = str2func("DDM_density"); 
    model.kappa = 5;
    model.mixture_weight = 0.98;
    model.p_0 = 1/(2*7);
    model.matching_parameters = str2func("Matching_Parameters_RegDDM"); 
    model.matching_gradients = str2func("Matching_Gradients_RegDDM"); 
    
    model.T_inv = str2func("T_inv");  % used in matching_function_1

%% Initialization for VB and MCMC:
% alpha = ( log(-v0_lure), log(v0_target), log(-v0_nontarget), log(sv0), 
%           log(a0-z0-sz0/2), log(z0-sz0/2), log(sz0), log(tau0 - stau0/2), log(stau0),
%           log(-v2_lure), log(v2_target), log(-v2_nontarget), log(sv2), 
%           log(a2-z2-sz2/2), log(z2-sz2/2), log(sz2), log(tau2 - stau2/2), log(stau2))

    minRT = min(data{1,1}.RT);
    for j = 2:J
        if minRT > min(data{j,1}.RT)
            minRT = min(data{j,1}.RT);
        end
    end
    v0_lure = 2*rand()-2;   v0_target = 2*rand()+1;  v0_nontarget = 2*rand()-3;
    sv0 = rand();
    a0 = 2*rand()+1;   z0 = a0/2;   sz0 = 0.5*rand();
    tau0 = 0.36; stau0 = minRT*rand();

    v2_lure = 2*rand()-2;   v2_target = 2*rand()+1;  v2_nontarget = 2*rand()-3;
    sv2 = rand();
    a2 = 2*rand()+1;   z2 = a2/2;   sz2 = 0.5*rand();
    tau2 = 0.36; stau2 = minRT*rand();

    initial_mu_alpha = [log(-v0_lure), log(v0_target), log(-v0_nontarget), log(sv0),... 
              log(a0-z0-sz0/2), log(z0-sz0/2), log(sz0), log(tau0 - stau0/2), log(stau0),...
              log(-v2_lure), log(v2_target), log(-v2_nontarget), log(sv2),... 
              log(a2-z2-sz2/2), log(z2-sz2/2), log(sz2), log(tau2 - stau2/2), log(stau2)];

    initial_Sigma_alpha = 0.1*eye(D_alpha);
    initial_alpha = initial_mu_alpha'*ones(1,J);

    initial_log_a_d = zeros(D_alpha,1);  

    initial_beta_vec = zeros(prod(model.beta_dim),1);
    if model.prior_Sigma_alpha == "Informative"
        VB_settings.Domain_Knowledge_initial = [initial_alpha(:); initial_mu_alpha(:)];
    elseif model.prior_Sigma_alpha == "Marginally noninformative"
        VB_settings.Domain_Knowledge_initial = [initial_alpha(:); initial_mu_alpha(:); initial_log_a_d];
    end

    MCMC_setting.initialization.beta_vec = initial_beta_vec;
    MCMC_setting.initialization.alpha = initial_alpha;
    MCMC_setting.initialization.mu = initial_mu_alpha; %the initial values for parameter \mu
    MCMC_setting.initialization.Sigma = initial_Sigma_alpha; % the initial values for \Sigma
    MCMC_setting.initialization.a_d = 1./random('gam',1/2,1,D_alpha,1);

