
J = length(data); 	

    model.structure = "Hierarchical"; % "Regression" or "Nonhierarchical" or "Hierarchical"
    model.prior_Sigma_alpha = "Marginally noninformative"; % "Informative" or "Marginally noninformative"
    model.subject_param_dim = 12;% number of random effects per participant (D_alpha)

    model.num_subjects = J; 
    D_alpha = model.subject_param_dim;
    
    model.kappa = 5;
    
    
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
    model.mixture_weight = 0.9999;
    model.p_0 = 1/(4);
    model.matching_parameters = str2func("Matching_Parameters_HDDM_Lexical");
    model.matching_gradients = str2func("Matching_Gradients_HDDM_Lexical");

    model.T_inv = str2func("T_inv_HDDM_Lexical");
    model.Stacking_obs = str2func("RT_Stacking_observations");
%% Initialization

minRT = min(data{1,1}.RT);
    for j = 2:J
        if minRT > min(data{j,1}.RT)
            minRT = min(data{j,1}.RT);
        end
    end
    mu_v = 10*rand(1,4) - 5; % -5 < mu_v < 5
    sv = 2*rand(); % 0 < sv < 2
    a = 1.5*rand(1,2) + 0.5; % 0.5 < a < 2
    mu_z = a/2;
    sz = 0.5*rand(); % 0 < sz < 0.5   
    mu_t0 = minRT/2; % 0.1 < t0 < 0.5, 0 < st0 < 0.2 
    st0 = minRT/2;
    
    initial_mu_alpha = [mu_v, log(sv), log(a - mu_z - 0.5*sz), log(mu_z - 0.5*sz), log(sz) ,...
            log(mu_t0 - 0.5*st0), log(st0) ]';
    initial_Sigma_alpha = 0.1*eye(D_alpha);
%     initial_Sigma_alpha = iwishrnd(model.prior_par.Psi,model.prior_par.v);
    initial_alpha = repmat(initial_mu_alpha,1,J);

    initial_log_a_d = zeros(D_alpha,1);  

    if model.prior_Sigma_alpha == "Informative"
        VB_settings.Domain_Knowledge_initial = [initial_alpha(:);  initial_mu_alpha(:)];
    elseif model.prior_Sigma_alpha == "Marginally noninformative"
        VB_settings.Domain_Knowledge_initial = [initial_alpha(:);  initial_mu_alpha(:); initial_log_a_d];
    end

    MCMC_setting.initialization.alpha = initial_alpha;
    MCMC_setting.initialization.mu = initial_mu_alpha; %the initial values for parameter \mu
    MCMC_setting.initialization.Sigma = initial_Sigma_alpha; % the initial values for \Sigma
    MCMC_setting.initialization.a_d = 1./random('gam',1/2,1,D_alpha,1);

