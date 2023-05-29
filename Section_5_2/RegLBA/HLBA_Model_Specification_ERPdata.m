
    J = length(data); 	
    model.num_subjects = J;     

    model.structure = "Regression"; % "Regression" or "Nonhierarchical" or "Hierarchical"
    model.prior_Sigma_alpha = "Informative"; % "Informative" or "Marginally noninformative"
    model.subject_param_dim = 10;% number of random effects per participant (D_alpha)
 
    model.density = str2func("LBA_pdf"); 
    model.mixture_weight = 0.98;
    model.p_0 = 1/(2*7);
    model.matching_parameters = str2func("HLBA_Matching_Parameters_ERPdata"); 
    model.matching_gradients = str2func("HLBA_Matching_Gradients_ERPdata"); 
    

%% Prior Specification: 
      
    D_alpha = model.subject_param_dim;
    model.beta_dim = [1,size(data{1,1}.X,2)];
    model.prior_par.mu_beta = zeros(prod(model.beta_dim),1);
    model.prior_par.Sigma_beta = 10*eye(prod(model.beta_dim));

    model.prior_par.mu = zeros(D_alpha,1); 
    model.prior_par.cov = 10*eye(D_alpha); 
    
    % hyperparameters for a noninformative prior for Sigma:
    model.prior_par.v_a = 2; 
    model.prior_par.A_d = ones(D_alpha,1);
    
    % hyperparameters for an informative prior for Sigma:
    model.prior_par.v = 20;    
    model.prior_par.Psi = eye(D_alpha);
    

%% Initialization for VB and MCMC:
% alpha = ( log(b-A), log(A), v_s^s, v_s^m, v_m^s, v_m^m, v_c, v_e, log(tau0), log(tau))

    minRT = min(data{1,1}.RT);
    for j = 2:J
        if minRT > min(data{j,1}.RT)
            minRT = min(data{j,1}.RT);
        end
    end
    b = 3 + rand();  A = 1.5 + rand();
    v_ss = 4.22 + rand();  v_sm = -2.72 + rand(); 
    v_ms = -1.3 + rand();    v_mm = 3.6 + rand();
    v_c = -0.036 + 0.1*rand();    
    v_e = 0.08 + 0.1*rand() ; % 0 < sv < 2
    
    tau0 = minRT/2; % 0.1 < t0 < 0.5, 0 < st0 < 0.2
    tau = 0.01*rand();
    initial_mu_alpha = [log(b-A), log(A), v_ss, v_sm, v_ms, v_mm, v_c, v_e, log(tau0), log(tau)]';

%     initial_Sigma_alpha = 0.1*eye(D_alpha);
    initial_Sigma_alpha = iwishrnd(model.prior_par.Psi,model.prior_par.v);
%     initial_alpha = mvnrnd(initial_mu_alpha',initial_Sigma_alpha,J)';
    initial_alpha = repmat(initial_mu_alpha,1,J);

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

