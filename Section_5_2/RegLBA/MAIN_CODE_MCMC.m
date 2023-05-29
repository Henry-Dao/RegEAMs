
%% Description:
% This is the main code for running MCMC for hierarchical EAMs, including LBA and DDMs.
% 
% Reference: https://arxiv.org/abs/2302.10389
% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com
%% Step 0: ----------------------- PREPERATION ----------------------------

    clear all; clc

    scaling_data = true;  scaled_std = 1; % scaling_data = true means the covariates are standardized
    
    load('ERP_data.mat') % load the data
    
    save_path = '';
    save_name = 'MCMC_result';   
  
%% Step 1: -------------------- Model specification ----------------------- 
    
    if scaling_data == true
        data = scale_data(data,scaled_std);
    end

    HLBA_Model_Specification_ERPdata
    J = length(data); % number of subjects
    p1 = D_alpha*(J + 1) + prod(model.beta_dim); % dim(theta_1) = p1; 
    
    
%% Step 2: ------------------------ MCMC Setting --------------------------

    PMwG = str2func("PMwG");
    MCMC_setting.CMC_alpha = str2func("CMC_alpha"); % name of the CMC_alpha function
    MCMC_setting.CMC_beta = str2func("CMC_beta"); % name of the CMC_alpha function

    MCMC_setting.when_to_reduce_the_numer_particles = 2000;
    MCMC_setting.CMC_alpha_number_particles = 500; % number of particles in the conditional Monte Carlo algorithm  
    MCMC_setting.CMC_alpha_number_particles_after = 150;
    MCMC_setting.CMC_alpha_w_mix = 0.95; % mixture weight in the initial proposal
    MCMC_setting.CMC_alpha_epsilon = 0.1; % the scaling parameter for the proposal during burn in and initial adaptation.
    MCMC_setting.CMC_alpha_w1 = 0.65;     MCMC_setting.CMC_alpha_w2 = 0.05; % mixture weights in the better proposal

    MCMC_setting.CMC_beta_number_particles = 500; % number of particles in the conditional Monte Carlo algorithm  
    MCMC_setting.CMC_beta_number_particles_after = 150;
    MCMC_setting.CMC_beta_w_mix = 0.95; % mixture weight in the initial proposal
    MCMC_setting.CMC_beta_epsilon = 0.01; % the scaling parameter for the proposal during burn in and initial adaptation.
    MCMC_setting.CMC_beta_w1 = 0.65;     MCMC_setting.CMC_beta_w2 = 0.05; % mixture weights in the better proposal

    MCMC_setting.burnin = 0;     % number of iterations in the burn-in stage
    MCMC_setting.adaptation_stage = 300; % number of iterations in the adaptation stage
    MCMC_setting.number_samples = 2000; % number of draws used to estimate the covariance matrix in the proposal    
    MCMC_setting.update_period = 20;  % update the proposal after 20 iterations    
    MCMC_setting.stop_update = 24999; % after N_threshold, stop updating the proposal 
    MCMC_setting.start_sampling = 1;    % we take the last 10000 draws    
    
    
    MCMC_setting.filtering_alpha = false;     % the burn in iterations
    MCMC_setting.stacking = false;     % stacking or for-loop
    
    J = length(data);
%     D_G = d_alpha*d + 2*D_alpha + D_alpha*(D_alpha+1)/2; 
    MCMC_setting.proposals_alpha = cell(J,1);
    for j = 1:J
        MCMC_setting.proposals_alpha{j,1}.better_proposal = false;
        MCMC_setting.proposals_alpha{j,1}.mean = [];
        MCMC_setting.proposals_alpha{j,1}.cov = [];
    end 
    MCMC_setting.proposals_beta.better_proposal = false;
    MCMC_setting.proposals_beta.mean = []; % assign value for preallocation, only use when switch to better proposal
    MCMC_setting.proposals_beta.cov = []; 
    

    MCMC_setting.save = false;
    MCMC_setting.save_period = 1000; % save after 1000 iterations
    MCMC_setting.save_name = save_name;
    MCMC_setting.save_path = save_path;
    
    MCMC_setting.display_iter = true;     
    MCMC_setting.display_iter_running_time = true;   
    MCMC_setting.display_warnings = true;  
    MCMC_setting.display_particles = true;  

%% Step 4: -------------------- Run the MCMC Sampler ----------------------

    MCMC_draws = PMwG(model,data,MCMC_setting);

%% Step 5: ------------------------ Save data -----------------------------

    save([save_path,save_name,'.mat'],'MCMC_draws','MCMC_setting','model');    
