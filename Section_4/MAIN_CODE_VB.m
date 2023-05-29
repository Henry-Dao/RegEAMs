
%% Description:
% This is the main code for running VB for hierarchical EAMs, including LBA and DDMs.
% 
% Reference: https://arxiv.org/abs/2302.10389
% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com
%% Step 0: ----------------------- PREPERATION ----------------------------
clear all; clc

    load('Simulated_data_medium.mat') % load the data
    
    save_path = '';
    save_name = 'VB_result';
  
%% Step 1: -------------------- Model specification ----------------------- 
    
    HDDM_Model_Specification_Lexical

    J = length(data); % number of subjects
    p1 = D_alpha*(J + 2); % dim(theta_1) = p1; 


%% Step 2: ------------------------- VB Setting ---------------------------
    VB_settings.Initialization_Strategy = "Domain Knowledge";   

    VB_settings.VAFC = str2func("Hybrid_VAFC"); 
    VB_settings.Likelihood = str2func("Likelihood_Hybrid"); 
    VB_settings.prior_density = str2func("prior_density_Hybrid"); 
    VB_settings.q_vb = str2func("q_VAFC"); 
    VB_settings.p_Sigma_alpha = str2func("p_Sigma_alpha");

    VB_settings.r = 40; % number of factors in VAFC    
    r = VB_settings.r; % total number of factors in VAFC 
    
    VB_settings.max_iter = 25000; % the total number of iterations
    VB_settings.min_iter = 100;
    VB_settings.max_norm = 5000; 
    VB_settings.gradient_clipping = true;
    
    VB_settings.I = 10; % number of Monte Carlo samples used to estimate 
    VB_settings.window = 100;    
    VB_settings.patience_parameter = 50;       
    VB_settings.learning_rate = "ADAM"; % "ADAM" or "ADADELTA"    
    VB_settings.ADADELTA.eps = 10^(-7); 
    VB_settings.ADADELTA.v = 0.95;     
    VB_settings.ADADELTA.eps = 10^(-7); 
    VB_settings.ADADELTA.E_g2 = zeros(p1*(r+2),1); 
    VB_settings.ADADELTA.E_delta2 = zeros(p1*(r+2),1); 
    VB_settings.ADAM.adapt_tau_1=0.9;
    VB_settings.ADAM.adapt_tau_2=0.99;
    VB_settings.ADAM.adapt_epsilon=10^-8;
    VB_settings.ADAM.adapt_alpha_mu=0.01;
    VB_settings.ADAM.adapt_alpha_B=0.001;
    VB_settings.ADAM.adapt_alpha_d=0.001;
    VB_settings.ADAM.m_t_mu = 0;
    VB_settings.ADAM.v_t_mu= 0;
    VB_settings.ADAM.m_t_B= 0;
    VB_settings.ADAM.v_t_B= 0;
    VB_settings.ADAM.m_t_d= 0;
    VB_settings.ADAM.v_t_d=0;

    VB_settings.threshold = 1; % threshold for convergence criterion
    VB_settings.silent = false; % display the estimates at each iteration
    VB_settings.display_period = 1;
    VB_settings.generated_samples = true; 

    VB_settings.store_lambdas = false;
    VB_settings.save = true;
    VB_settings.save_period = 1000; % save after 1000 iterations
    VB_settings.save_name = save_name;
    VB_settings.save_path = save_path;
     
%% Step 3: ---------------------- VB Initialization -----------------------
    tic
    if VB_settings.Initialization_Strategy == "Domain Knowledge"
        % ===================== METHOD 1: MANUALLY CHOSEN VALUES ==================

        %  Now is put inside the Model_Specification script, just run the code
        %  below
        initial = VB_settings.Domain_Knowledge_initial;
   
        % ========================== METHOD 5: MLE ===========================
    elseif VB_settings.Initialization_Strategy == "MLE"
        MLE_model = model;
        MLE_model.objective_function = str2func("minus_log_hddm_density");
        mle = zeros(D_alpha,J);
        log_likelihood = zeros(J,1);
        parfor j = 1:J
	        disp([' finding MLE of subject ',num2str(j)])
	        subject_j = data{j,1};
            x0 = initial_alpha(:,j)';
            options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true);
            obj_fun = @(x)MLE_model.objective_function(x,subject_j,MLE_model);
            [mle(:,j),log_likelihood(j)] = fminunc(obj_fun,x0,options);
        end
        alpha_mle = mle(1:D_alpha,:);
        VB_settings.MLE_initialization.model = MLE_model;
    	initial = [alpha_mle(:); mean(alpha_mle,2); zeros(D_alpha,1)];
        VB_settings.MLE_initialization.alpha = alpha_mle;
       
        VB_settings.MLE_initialization.log_likelihood = log_likelihood;
    end
% *************************************************************************
initial_running_time = toc;
disp(['The initialization time is ',num2str(round(initial_running_time/60,1)),' minutes']) 
lambda.mu = initial;
lambda.B = zeros(p1,VB_settings.r)/VB_settings.r;    lambda.B = tril(lambda.B);
lambda.d = 0.01*ones(p1,1);       
VB_settings.initial = lambda;
save([save_path,save_name,'.mat'],'VB_settings','model'); 
%% Step 4: ------------------------ VB algorithm --------------------------
disp(' ~~~~~~~~~~~~~ Running VB algorithm ~~~~~~~~~~~~~ ')

    VB_results = VB_settings.VAFC(model,data,VB_settings);   

VB_results.initial_running_time = initial_running_time;
%% Step 5:  -------------------- Extract the results  ---------------------
disp(['The running time is ',num2str(round(VB_results.running_time/60,1)),' minutes'])      
disp(['The initialization time is ',num2str(round(initial_running_time/60,1)),' minutes'])   
plot(VB_results.LB_smooth)
title('Smoothed lower bound estimates')
                                                                          
%% Step 6:  --------------------- Save the results  ----------------------
save([save_path,save_name,'.mat'],'VB_results','VB_settings','model'); 