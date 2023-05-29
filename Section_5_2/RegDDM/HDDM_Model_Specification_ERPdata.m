
    J = length(data); 	
    model.num_subjects = J;     
    

    model.structure = "Regression"; % "Regression" or "Nonhierarchical" or "Hierarchical"
    model.prior_Sigma_alpha = "Informative"; % "Informative" or "Marginally noninformative"
    model.subject_param_dim = 9;% number of random effects per participant (D_alpha)

    model.num_random_effects = 9;% number of random effects per participant (D_alpha)

    model.alpha_index.v0 = 1;   model.alpha_index.v = 2;    model.alpha_index.sv = 3;
    model.alpha_index.azsz = 4;   model.alpha_index.zsz = 5;    model.alpha_index.sz = 6; 
    model.alpha_index.tau0 = 7;   model.alpha_index.tau = 8;     model.alpha_index.stau = 9;


    model.u_index = {1, 2, 3, 4, 5, 6, 7}; 

    
    model.kappa = 5;

% ----------------------------------------------------------
    model.beta_dim(1) = 1; 
    model.beta_dim(2) = size(data{1,1}.X,2);% number of random effects per participant
    
    D_alpha = model.num_random_effects;
    d = model.beta_dim(2);    
    d_alpha = model.beta_dim(1); % number of linked parameters
    % beta is matrix of dimension (d_alpha x d)
    
    model.prior_par.mu = zeros(D_alpha,1); 
    model.prior_par.cov = eye(D_alpha); 
    
    % hyperparameters for a noninformative prior for Sigma:
    model.prior_par.v_a = 2; 
    model.prior_par.A_d = ones(D_alpha,1);
    
    % hyperparameters for an informative prior for Sigma:
    model.prior_par.v = 20;    
    model.prior_par.Psi = eye(D_alpha);
    
    model.prior_par.mu_beta = zeros(d_alpha*d,1);
    model.prior_par.Sigma_beta = eye(d_alpha*d); 
      
    model.density = str2func("DDM_density"); 
    model.kappa = 5;
    model.mixture_weight = 0.98;
    model.p_0 = 1/(2*7);
    model.Stacking_obs = str2func("ERPdata_Stacking_observations");
    model.matching_parameters = str2func("HDDM_Matching_Parameters_ERPdata"); % used in PMwG and VB
    model.matching_gradients = str2func("HDDM_Matching_Gradients_ERPdata"); % only used in VB
    model.T_inv = str2func("T_inv_HDDM");  % used in matching_function_1
    
%% Stacking data
data_stacked = [];

%% Initialization for VB and MCMC:
% alpha = ( v0_s, v_s, v0_m, v_m, log(sv), log(a-z-sz/2), log(z-sz/2), log(sz),
%         log(tau^0 - stau/2), log(tau), log(stau) )

    minRT = min(data{1,1}.RT);
    for j = 2:J
        if minRT > min(data{j,1}.RT)
            minRT = min(data{j,1}.RT);
        end
    end
    v0 = 3+ rand();     v = 0.01*rand();
    sv = 2*rand(); % 0 < sv < 2
    a = 1.5*rand() + 0.5; % 0.5 < a < 2
    mu_z = a/2;
    sz = 0.5*rand(); % 0 < sz < 0.5

    st0 = 0.1*rand();
    tau = 0.01*rand();
    tau0 = minRT + st0/2 + 0.1*rand(); % 0.1 < t0 < 0.5, 0 < st0 < 0.2

    initial_mu_alpha = [v0, v, log(sv),  log(a - mu_z - 0.5*sz), log(mu_z - 0.5*sz), ...
        log(sz) , log(tau0 - 0.5*st0), log(tau), log(st0) ]';

    initial_Sigma_alpha = 0.1*eye(D_alpha);
    initial_alpha = repmat(initial_mu_alpha,1,J);

    beta_matrix = zeros(d_alpha,d);
    initial_beta_vec = beta_matrix(:);  
    initial_log_a_d = zeros(D_alpha,1);  

    
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

