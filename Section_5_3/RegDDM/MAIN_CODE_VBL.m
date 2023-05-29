%% Description:
% This is the main code for running VBL for RegDDM.
%
% Reference: https://arxiv.org/abs/2302.10389
% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com
%% Step 0: ----------------------- PREPERATION ----------------------------
clear all; clc
    rng(425);    
    save_path = '';
    save_name = 'VB_result';

%% Step 1: -------------------- Model specification -----------------------
    load('HCP_fulldata_combined.mat') % load the data
    data = scale_data(data,1); % standardize the covariates
    
    HDDM_Model_Specification
    
    
%% Step 2: ------------------------- VB Setting ---------------------------

VB_settings.Initialization_Strategy = "Domain Knowledge";  
VB_settings.Likelihood = str2func("Likelihood_Hybrid_informativeprior_Sigma");
VB_settings.prior_density = str2func("prior_density_Hybrid_informativeprior_Sigma");
VB_settings.q_vb = str2func("q_VAFC_separate");
VB_settings.p_Sigma_alpha = str2func("p_Sigma_alpha_informativeprior_Sigma");

VB_settings.r = 4; % number of factors in VAFC
VB_settings.max_iter = 50000; % the total number of iterations
VB_settings.min_iter = 50;
VB_settings.max_norm = 100000;

VB_settings.I = 1; % number of Monte Carlo samples used to estimate
VB_settings.window = 100;
VB_settings.patience_parameter = 50;
VB_settings.learning_rate.v = 0.95;
VB_settings.learning_rate.eps = 10^(-7);

VB_settings.learning_rate.adapt_tau_1=0.9;
VB_settings.learning_rate.adapt_tau_2=0.99;
VB_settings.learning_rate.adapt_epsilon=10^-8;
VB_settings.learning_rate.adapt_alpha_mu=0.01;
VB_settings.learning_rate.adapt_alpha_B=0.001;
VB_settings.learning_rate.adapt_alpha_d=0.001;

VB_settings.threshold = 10000; % threshold for convergence criterion
VB_settings.silent = false; % display the estimates at each iteration
VB_settings.display_period = 1;
VB_settings.generated_samples = false;

VB_settings.store_lambdas = false;
VB_settings.save = true;
VB_settings.save_period = 1000; % save after 1000 iterations
VB_settings.save_name = save_name;
VB_settings.save_path = save_path;
%% Step 3: ---------------------- VB Initialization -----------------------
J = length(data); % number of subjects
p1 = D_alpha*(J + 1); % dim(theta_1) = p1;
r = VB_settings.r;
tic  
if VB_settings.Initialization_Strategy == "Domain Knowledge"
    % ===================== METHOD 1: MANUALLY CHOSEN VALUES ==================

    initial = VB_settings.Domain_Knowledge_initial;
    for j=1:J
        lambda.mu{j,1} = initial_alpha(:,j);
        lambda.B{j,1} = 0.01*ones(D_alpha,r)/r;
        lambda.d{j,1} = 0.01*ones(D_alpha,1);
    end
    num_covariates = model.beta_dim(2);
    d_alpha = model.beta_dim(1);
    lambda.mu{J+1,1}=[mean(initial_alpha,2); zeros(prod(model.beta_dim),1)];
    lambda.B{J+1,1} = 0.01*ones(D_alpha+prod(model.beta_dim),r)/r;
    lambda.d{J+1,1} = 0.01*ones(D_alpha+prod(model.beta_dim),1);
    VB_settings.initial = lambda;
    % ========================== METHOD 2: MLE ===========================
elseif VB_settings.Initialization_Strategy == "MLE"
    MLE_model = model;
    MLE_model.matching_parameters = str2func("Matching_Parameters_RegLBA");
    MLE_model.matching_gradients = str2func("Matching_Gradients_RegLBA");
    MLE_model.objective_function = str2func("minus_log_hddm_nobeta_density");
    mle = zeros(D_alpha,J);
        log_likelihood = zeros(J,1);
        for j = 1:J
            disp([' finding MLE of subject ',num2str(j)])
            subject_j = data{j,1};
            x0 = initial_alpha(:,j)';
            options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true);
            obj_fun = @(x)MLE_model.objective_function(x,subject_j,MLE_model);
            [mle(:,j),log_likelihood(j)] = fminunc(obj_fun,x0,options);
        end
        alpha_mle = mle(1:D_alpha,:);

    VB_settings.MLE_initialization.model = MLE_model;
    initial = [alpha_mle(:); mean(alpha_mle,2)];
    for j=1:J
        lambda.mu{j,1} = alpha_mle(:,j);
        lambda.B{j,1} = 0.01*ones(D_alpha,r)/r;
        lambda.d{j,1} = 0.01*ones(D_alpha,1);
    end
    num_covariates = model.beta_dim(2);
    d_alpha = model.beta_dim(1);
    lambda.mu{J+1,1}=[mean(alpha_mle,2); zeros(prod(model.beta_dim),1)];
    lambda.B{J+1,1} = 0.01*ones(D_alpha+prod(model.beta_dim),r)/r;
    lambda.d{J+1,1} = 0.01*ones(D_alpha+prod(model.beta_dim),1);
    VB_settings.initial = lambda;

    VB_settings.MLE_initialization.alpha = alpha_mle;
    VB_settings.MLE_initialization.log_likelihood = log_likelihood;
end
% *************************************************************************
initial_running_time = toc;
disp(['The initialization time is ',num2str(round(initial_running_time/60,1)),' minutes'])



%% Step 4: ------------------------ VB algorithm --------------------------
disp(' ~~~~~~~~~~~~~ Running VB algorithm ~~~~~~~~~~~~~ ')

    for k=1:J+1
        m_t_mu{k,1}=0;
        v_t_mu{k,1}=0;

        m_t_B{k,1}=0;
        v_t_B{k,1}=0;

        m_t_d{k,1}=0;
        v_t_d{k,1}=0;
    end
    Likelihood = VB_settings.Likelihood;
    prior_density = VB_settings.prior_density;
    p_Sigma_alpha = VB_settings.p_Sigma_alpha;
    q_vb = VB_settings.q_vb;

    lambda = VB_settings.initial;
    I = VB_settings.I;
    r = VB_settings.r;
    max_iter = VB_settings.max_iter;
    max_norm = VB_settings.max_norm;    
    window = VB_settings.window;
    patience_parameter = VB_settings.patience_parameter;   
    
    if VB_settings.store_lambdas == true
        lambda_store = cell(max_iter,1);
        lambda_store{1,1} = lambda;
    end
    LB = zeros(max_iter,1);
    J = model.num_subjects;
    D_alpha = sum(model.subject_param_dim);
    p = D_alpha*(J + 1) + d_alpha*num_covariates;
    
    Sigma_Psi = model.prior_par.Psi;
    df = model.prior_par.v + D_alpha + J + 1;
    
    v = VB_settings.learning_rate.v; eps = VB_settings.learning_rate.eps; 
    E_g2 = zeros(p*(r+2),1); E_delta2 = zeros(p*(r+2),1);
    iter_time = toc;
    for t = 1:window
        mu = lambda.mu;
        B = lambda.B;
        d = lambda.d;

        for i = 1:I
            % (1) Generate theta
            for k=1:J
                epsilon{k,1}= randn(D_alpha,1);
                z{k,1}=randn(r,1);
                ALPHA(:,k) = mu{k,1} + B{k,1}*z{k,1} + d{k,1}.*epsilon{k,1};

            end

            epsilon{J+1,1}=randn(D_alpha+d_alpha*num_covariates,1);
            z{J+1,1}=randn(r,1);
            beta_vec_mu_alpha=mu{J+1,1}+B{J+1,1}*z{J+1,1} + d{J+1,1}.*epsilon{J+1,1};
            beta_vec = beta_vec_mu_alpha(1 : d_alpha*num_covariates);
            mu_alpha = beta_vec_mu_alpha(d_alpha*num_covariates + 1:end);


            Psi = Sigma_Psi;
            for j=1:J
                Psi = Psi + (ALPHA(:,j)-mu_alpha)*(ALPHA(:,j)-mu_alpha)';
            end
            Sigma_alpha = iwishrnd(Psi,df);
            % (2) Calculate the likelihood, prior and q_vb

            like = Likelihood(model,data,ALPHA,beta_vec); % last argument = "true" means to compute the gradients
            prior = prior_density(model,beta_vec,ALPHA,mu_alpha,Sigma_alpha);
            q_lambda = q_vb(ALPHA,beta_vec_mu_alpha,mu,B,d,J);
            q_Sigma = p_Sigma_alpha(Sigma_alpha,df,Psi,ALPHA,mu_alpha,num_covariates,d_alpha);

            % (3) Estimate the lower bound
            LBs(i) = like.log +  prior.log - q_lambda.log - q_Sigma.log;
            lik_grad_reshape = reshape(like.grad(1:D_alpha*J),D_alpha,J);
            prior_grad_reshape = reshape(prior.grad(1:D_alpha*J),D_alpha,J);
            q_Sigma_grad_reshape = reshape(q_Sigma.grad(1:D_alpha*J),D_alpha,J);
            % (4) Cumpute the gradients
            for k=1:J
                grad_theta_1_LB = lik_grad_reshape(:,k) + prior_grad_reshape(:,k) - q_lambda.grad{k,1} - q_Sigma_grad_reshape(:,k);
                temp = grad_theta_1_LB;
                gradmu{k,1}(:,i) = temp;
                gradB{k,1}(:,:,i) = temp*z{k,1}';
                gradd{k,1}(:,i) = temp.*epsilon{k,1};

            end
            k = J + 1;
            lik_grad_reshape = reshape(like.grad(1+D_alpha*J:end),D_alpha + d_alpha*num_covariates,1);
            prior_grad_reshape = reshape(prior.grad(1+D_alpha*J:end),D_alpha + d_alpha*num_covariates,1);
            q_Sigma_grad_reshape = reshape(q_Sigma.grad(1+D_alpha*J:end),D_alpha + d_alpha*num_covariates,1);
            grad_theta_1_LB = lik_grad_reshape + prior_grad_reshape - q_lambda.grad{k,1} - q_Sigma_grad_reshape;
            temp = grad_theta_1_LB;
            gradmu{k,1}(:,i) = temp;
            gradB{k,1}(:,:,i) = temp*z{k,1}';
            gradd{k,1}(:,i) = temp.*epsilon{k,1};
        end


        % Estimate the gradients
        for k=1:J+1
            grad_mu{k,1} = mean(gradmu{k,1},2);
            grad_B{k,1} = mean(gradB{k,1},3);
            grad_D{k,1} = mean(gradd{k,1},2);
        end
        %g = [grad_mu;grad_B(:);grad_D]; % Stack gradient of LB into 1 column

        for k=1:J
            m_t_mu{k,1}=VB_settings.learning_rate.adapt_tau_1*m_t_mu{k,1}+(1-VB_settings.learning_rate.adapt_tau_1)*grad_mu{k,1};
            v_t_mu{k,1}=VB_settings.learning_rate.adapt_tau_2*v_t_mu{k,1}+(1-VB_settings.learning_rate.adapt_tau_2)*(grad_mu{k,1}.^2);
            mt_hat_mu{k,1}=m_t_mu{k,1}./(1-(VB_settings.learning_rate.adapt_tau_1.^t));
            vt_hat_mu{k,1}=v_t_mu{k,1}./(1-(VB_settings.learning_rate.adapt_tau_2.^t));
            temp_mu{k,1} = lambda.mu{k,1}+VB_settings.learning_rate.adapt_alpha_mu*(mt_hat_mu{k,1}./(sqrt(vt_hat_mu{k,1})+VB_settings.learning_rate.adapt_epsilon));
            if sum(isnan(temp_mu{k,1}))>0
                lambda.mu{k,1}=lambda.mu{k,1};
            else
                lambda.mu{k,1}=temp_mu{k,1};
            end

            m_t_B{k,1}=VB_settings.learning_rate.adapt_tau_1*m_t_B{k,1}+(1-VB_settings.learning_rate.adapt_tau_1)*grad_B{k,1}(:);
            v_t_B{k,1}=VB_settings.learning_rate.adapt_tau_2*v_t_B{k,1}+(1-VB_settings.learning_rate.adapt_tau_2)*(grad_B{k,1}(:).^2);
            mt_hat_B{k,1}=m_t_B{k,1}./(1-(VB_settings.learning_rate.adapt_tau_1.^t));
            vt_hat_B{k,1}=v_t_B{k,1}./(1-(VB_settings.learning_rate.adapt_tau_2.^t));
            temp_B{k,1} = lambda.B{k,1}(:) + VB_settings.learning_rate.adapt_alpha_B*(mt_hat_B{k,1}./(sqrt(vt_hat_B{k,1})+VB_settings.learning_rate.adapt_epsilon));
            if sum(isnan(temp_B{k,1}))>0
                temp_B_VB_beta{k,1}=lambda.B{k,1}(:);
            else
                temp_B_VB_beta{k,1}=temp_B{k,1};
            end

            lambda.B{k,1} = reshape(temp_B_VB_beta{k,1},D_alpha,r);
            lambda.B{k,1} = tril(lambda.B{k,1});

            m_t_d{k,1}=VB_settings.learning_rate.adapt_tau_1*m_t_d{k,1}+(1-VB_settings.learning_rate.adapt_tau_1)*grad_D{k,1};
            v_t_d{k,1}=VB_settings.learning_rate.adapt_tau_2*v_t_d{k,1}+(1-VB_settings.learning_rate.adapt_tau_2)*(grad_D{k,1}.^2);
            mt_hat_d{k,1}=m_t_d{k,1}./(1-(VB_settings.learning_rate.adapt_tau_1.^t));
            vt_hat_d{k,1}=v_t_d{k,1}./(1-(VB_settings.learning_rate.adapt_tau_2.^t));
            temp_d{k,1} = lambda.d{k,1}+VB_settings.learning_rate.adapt_alpha_d*(mt_hat_d{k,1}./(sqrt(vt_hat_d{k,1})+VB_settings.learning_rate.adapt_epsilon));
            if sum(isnan(temp_d{k,1}))>0
                lambda.d{k,1}=lambda.d{k,1};
            else
                lambda.d{k,1}=temp_d{k,1};
            end

        end
        k = J + 1;
            m_t_mu{k,1}=VB_settings.learning_rate.adapt_tau_1*m_t_mu{k,1}+(1-VB_settings.learning_rate.adapt_tau_1)*grad_mu{k,1};
            v_t_mu{k,1}=VB_settings.learning_rate.adapt_tau_2*v_t_mu{k,1}+(1-VB_settings.learning_rate.adapt_tau_2)*(grad_mu{k,1}.^2);
            mt_hat_mu{k,1}=m_t_mu{k,1}./(1-(VB_settings.learning_rate.adapt_tau_1.^t));
            vt_hat_mu{k,1}=v_t_mu{k,1}./(1-(VB_settings.learning_rate.adapt_tau_2.^t));
            temp_mu{k,1} = lambda.mu{k,1}+VB_settings.learning_rate.adapt_alpha_mu*(mt_hat_mu{k,1}./(sqrt(vt_hat_mu{k,1})+VB_settings.learning_rate.adapt_epsilon));
            if sum(isnan(temp_mu{k,1}))>0
                lambda.mu{k,1}=lambda.mu{k,1};
            else
                lambda.mu{k,1}=temp_mu{k,1};
            end

            m_t_B{k,1}=VB_settings.learning_rate.adapt_tau_1*m_t_B{k,1}+(1-VB_settings.learning_rate.adapt_tau_1)*grad_B{k,1}(:);
            v_t_B{k,1}=VB_settings.learning_rate.adapt_tau_2*v_t_B{k,1}+(1-VB_settings.learning_rate.adapt_tau_2)*(grad_B{k,1}(:).^2);
            mt_hat_B{k,1}=m_t_B{k,1}./(1-(VB_settings.learning_rate.adapt_tau_1.^t));
            vt_hat_B{k,1}=v_t_B{k,1}./(1-(VB_settings.learning_rate.adapt_tau_2.^t));
            temp_B{k,1} = lambda.B{k,1}(:) + VB_settings.learning_rate.adapt_alpha_B*(mt_hat_B{k,1}./(sqrt(vt_hat_B{k,1})+VB_settings.learning_rate.adapt_epsilon));
            if sum(isnan(temp_B{k,1}))>0
                temp_B_VB_beta{k,1}=lambda.B{k,1}(:);
            else
                temp_B_VB_beta{k,1}=temp_B{k,1};
            end

            lambda.B{k,1} = reshape(temp_B_VB_beta{k,1},D_alpha + d_alpha*num_covariates,r);
            lambda.B{k,1} = tril(lambda.B{k,1});

            m_t_d{k,1}=VB_settings.learning_rate.adapt_tau_1*m_t_d{k,1}+(1-VB_settings.learning_rate.adapt_tau_1)*grad_D{k,1};
            v_t_d{k,1}=VB_settings.learning_rate.adapt_tau_2*v_t_d{k,1}+(1-VB_settings.learning_rate.adapt_tau_2)*(grad_D{k,1}.^2);
            mt_hat_d{k,1}=m_t_d{k,1}./(1-(VB_settings.learning_rate.adapt_tau_1.^t));
            vt_hat_d{k,1}=v_t_d{k,1}./(1-(VB_settings.learning_rate.adapt_tau_2.^t));
            temp_d{k,1} = lambda.d{k,1}+VB_settings.learning_rate.adapt_alpha_d*(mt_hat_d{k,1}./(sqrt(vt_hat_d{k,1})+VB_settings.learning_rate.adapt_epsilon));
            if sum(isnan(temp_d{k,1}))>0
                lambda.d{k,1}=lambda.d{k,1};
            else
                lambda.d{k,1}=temp_d{k,1};
            end
     


        if VB_settings.store_lambdas == true
            lambda_store{t+1,1} = lambda;
        end
        % Estimate the Lower Bound
        LB(t) = mean(LBs);
        if VB_settings.silent == false && (mod(t,VB_settings.display_period)==0)
            disp(['                 iteration ',num2str(t),' || LB: ',num2str(round(LB(t),1), '%0.1f'),...
                ' || standard error: ',num2str(round(std(LBs),2), '%0.2f'),' || running time: ',num2str(round(toc - iter_time,2)),' seconds']);
        end
        iter_time = toc;
    end

    %% Second stage: 
    
    LB_smooth = zeros(max_iter-window+1,1);
    LB_smooth(1) = mean(LB(1:t)); patience = 0; 
    lambda_best = lambda;
    max_best = LB_smooth(1);
    stopping_rule = "false";
    converge = "reach_max_iter";
    t = t + 1;
    t_smooth = 2; % iteration index for LB_smooth
    while t<= max_iter && stopping_rule == "false"
        mu = lambda.mu;
        B = lambda.B;
        d = lambda.d;

        for i = 1:I
            % (1) Generate theta
            for k=1:J
                epsilon{k,1}= randn(D_alpha,1);
                z{k,1}=randn(r,1);
                ALPHA(:,k) = mu{k,1} + B{k,1}*z{k,1} + d{k,1}.*epsilon{k,1};

            end

            epsilon{J+1,1}=randn(D_alpha+d_alpha*num_covariates,1);
            z{J+1,1}=randn(r,1);
            beta_vec_mu_alpha=mu{J+1,1}+B{J+1,1}*z{J+1,1} + d{J+1,1}.*epsilon{J+1,1};
            beta_vec = beta_vec_mu_alpha(1 : d_alpha*num_covariates);
            mu_alpha = beta_vec_mu_alpha(d_alpha*num_covariates + 1:end);


            Psi = Sigma_Psi;
            for j=1:J
                Psi = Psi + (ALPHA(:,j)-mu_alpha)*(ALPHA(:,j)-mu_alpha)';
            end
            Sigma_alpha = iwishrnd(Psi,df);
            % (2) Calculate the likelihood, prior and q_vb

            like = Likelihood(model,data,ALPHA,beta_vec); % last argument = "true" means to compute the gradients
            prior = prior_density(model,beta_vec,ALPHA,mu_alpha,Sigma_alpha);
            q_lambda = q_vb(ALPHA,beta_vec_mu_alpha,mu,B,d,J);
            q_Sigma = p_Sigma_alpha(Sigma_alpha,df,Psi,ALPHA,mu_alpha,num_covariates,d_alpha);

            % (3) Estimate the lower bound
            LBs(i) = like.log +  prior.log - q_lambda.log - q_Sigma.log;
            lik_grad_reshape = reshape(like.grad(1:D_alpha*J),D_alpha,J);
            prior_grad_reshape = reshape(prior.grad(1:D_alpha*J),D_alpha,J);
            q_Sigma_grad_reshape = reshape(q_Sigma.grad(1:D_alpha*J),D_alpha,J);
            % (4) Cumpute the gradients
            for k=1:J
                grad_theta_1_LB = lik_grad_reshape(:,k) + prior_grad_reshape(:,k) - q_lambda.grad{k,1} - q_Sigma_grad_reshape(:,k);
                temp = grad_theta_1_LB;
                gradmu{k,1}(:,i) = temp;
                gradB{k,1}(:,:,i) = temp*z{k,1}';
                gradd{k,1}(:,i) = temp.*epsilon{k,1};

            end
            k = J + 1;
            lik_grad_reshape = reshape(like.grad(1+D_alpha*J:end),D_alpha + d_alpha*num_covariates,1);
            prior_grad_reshape = reshape(prior.grad(1+D_alpha*J:end),D_alpha + d_alpha*num_covariates,1);
            q_Sigma_grad_reshape = reshape(q_Sigma.grad(1+D_alpha*J:end),D_alpha + d_alpha*num_covariates,1);
            grad_theta_1_LB = lik_grad_reshape + prior_grad_reshape - q_lambda.grad{k,1} - q_Sigma_grad_reshape;
            temp = grad_theta_1_LB;
            gradmu{k,1}(:,i) = temp;
            gradB{k,1}(:,:,i) = temp*z{k,1}';
            gradd{k,1}(:,i) = temp.*epsilon{k,1};
        end

        % Estimate the Lower Bound
        idx = isinf(LBs);
        if (sum(idx) >0)
            disp(['There are ',num2str(sum(idx)),' Inf LBs'])
        end
        LB(t) = mean(LBs(~idx));

        LB_smooth(t_smooth) = mean(LB(t-window+1:t));

        % Stopping Rule:
        if (LB_smooth(t_smooth)< max_best) || (abs(LB_smooth(t_smooth)-LB_smooth(t_smooth - 1))<0.00001)
            patience = patience + 1;
        else
            patience = 0;
            lambda_best = lambda;
            max_best = LB_smooth(t_smooth);
        end
        if (patience>patience_parameter) && (t>VB_settings.min_iter)
            stopping_rule = "true";
            if std(LB_smooth((t_smooth - patience_parameter + 1): t_smooth)) > VB_settings.threshold
                disp(['    Warning: VB might not converge to a good local mode',...
                    '(Initial LB = ',num2str(round(LB(1),1), '%0.1f'),...
                    ' || max LB = ',num2str( round(max_best,1), '%0.1f'),')']);
                converge = "no";
            else
                converge = "yes";
            end
            t_smooth = t_smooth + 1;
            break
        end

        % Estimate the gradients
        for k=1:J+1
            grad_mu{k,1} = mean(gradmu{k,1},2);
            grad_B{k,1} = mean(gradB{k,1},3);
            grad_D{k,1} = mean(gradd{k,1},2);
        end

        if t>20000
           VB_settings.learning_rate.adapt_alpha_mu=0.001;
        end
        
        
        for k=1:J
            m_t_mu{k,1}=VB_settings.learning_rate.adapt_tau_1*m_t_mu{k,1}+(1-VB_settings.learning_rate.adapt_tau_1)*grad_mu{k,1};
            v_t_mu{k,1}=VB_settings.learning_rate.adapt_tau_2*v_t_mu{k,1}+(1-VB_settings.learning_rate.adapt_tau_2)*(grad_mu{k,1}.^2);
            mt_hat_mu{k,1}=m_t_mu{k,1}./(1-(VB_settings.learning_rate.adapt_tau_1.^t));
            vt_hat_mu{k,1}=v_t_mu{k,1}./(1-(VB_settings.learning_rate.adapt_tau_2.^t));
            temp_mu{k,1} = lambda.mu{k,1}+VB_settings.learning_rate.adapt_alpha_mu*(mt_hat_mu{k,1}./(sqrt(vt_hat_mu{k,1})+VB_settings.learning_rate.adapt_epsilon));
            if sum(isnan(temp_mu{k,1}))>0
                lambda.mu{k,1}=lambda.mu{k,1};
            else
                lambda.mu{k,1}=temp_mu{k,1};
            end

            m_t_B{k,1}=VB_settings.learning_rate.adapt_tau_1*m_t_B{k,1}+(1-VB_settings.learning_rate.adapt_tau_1)*grad_B{k,1}(:);
            v_t_B{k,1}=VB_settings.learning_rate.adapt_tau_2*v_t_B{k,1}+(1-VB_settings.learning_rate.adapt_tau_2)*(grad_B{k,1}(:).^2);
            mt_hat_B{k,1}=m_t_B{k,1}./(1-(VB_settings.learning_rate.adapt_tau_1.^t));
            vt_hat_B{k,1}=v_t_B{k,1}./(1-(VB_settings.learning_rate.adapt_tau_2.^t));
            temp_B{k,1} = lambda.B{k,1}(:) + VB_settings.learning_rate.adapt_alpha_B*(mt_hat_B{k,1}./(sqrt(vt_hat_B{k,1})+VB_settings.learning_rate.adapt_epsilon));
            if sum(isnan(temp_B{k,1}))>0
                temp_B_VB_beta{k,1}=lambda.B{k,1}(:);
            else
                temp_B_VB_beta{k,1}=temp_B{k,1};
            end

            lambda.B{k,1} = reshape(temp_B_VB_beta{k,1},D_alpha,r);
            lambda.B{k,1} = tril(lambda.B{k,1});

            m_t_d{k,1}=VB_settings.learning_rate.adapt_tau_1*m_t_d{k,1}+(1-VB_settings.learning_rate.adapt_tau_1)*grad_D{k,1};
            v_t_d{k,1}=VB_settings.learning_rate.adapt_tau_2*v_t_d{k,1}+(1-VB_settings.learning_rate.adapt_tau_2)*(grad_D{k,1}.^2);
            mt_hat_d{k,1}=m_t_d{k,1}./(1-(VB_settings.learning_rate.adapt_tau_1.^t));
            vt_hat_d{k,1}=v_t_d{k,1}./(1-(VB_settings.learning_rate.adapt_tau_2.^t));
            temp_d{k,1} = lambda.d{k,1}+VB_settings.learning_rate.adapt_alpha_d*(mt_hat_d{k,1}./(sqrt(vt_hat_d{k,1})+VB_settings.learning_rate.adapt_epsilon));
            if sum(isnan(temp_d{k,1}))>0
                lambda.d{k,1}=lambda.d{k,1};
            else
                lambda.d{k,1}=temp_d{k,1};
            end

        end
        k = J + 1;
            m_t_mu{k,1}=VB_settings.learning_rate.adapt_tau_1*m_t_mu{k,1}+(1-VB_settings.learning_rate.adapt_tau_1)*grad_mu{k,1};
            v_t_mu{k,1}=VB_settings.learning_rate.adapt_tau_2*v_t_mu{k,1}+(1-VB_settings.learning_rate.adapt_tau_2)*(grad_mu{k,1}.^2);
            mt_hat_mu{k,1}=m_t_mu{k,1}./(1-(VB_settings.learning_rate.adapt_tau_1.^t));
            vt_hat_mu{k,1}=v_t_mu{k,1}./(1-(VB_settings.learning_rate.adapt_tau_2.^t));
            temp_mu{k,1} = lambda.mu{k,1}+VB_settings.learning_rate.adapt_alpha_mu*(mt_hat_mu{k,1}./(sqrt(vt_hat_mu{k,1})+VB_settings.learning_rate.adapt_epsilon));
            if sum(isnan(temp_mu{k,1}))>0
                lambda.mu{k,1}=lambda.mu{k,1};
            else
                lambda.mu{k,1}=temp_mu{k,1};
            end

            m_t_B{k,1}=VB_settings.learning_rate.adapt_tau_1*m_t_B{k,1}+(1-VB_settings.learning_rate.adapt_tau_1)*grad_B{k,1}(:);
            v_t_B{k,1}=VB_settings.learning_rate.adapt_tau_2*v_t_B{k,1}+(1-VB_settings.learning_rate.adapt_tau_2)*(grad_B{k,1}(:).^2);
            mt_hat_B{k,1}=m_t_B{k,1}./(1-(VB_settings.learning_rate.adapt_tau_1.^t));
            vt_hat_B{k,1}=v_t_B{k,1}./(1-(VB_settings.learning_rate.adapt_tau_2.^t));
            temp_B{k,1} = lambda.B{k,1}(:) + VB_settings.learning_rate.adapt_alpha_B*(mt_hat_B{k,1}./(sqrt(vt_hat_B{k,1})+VB_settings.learning_rate.adapt_epsilon));
            if sum(isnan(temp_B{k,1}))>0
                temp_B_VB_beta{k,1}=lambda.B{k,1}(:);
            else
                temp_B_VB_beta{k,1}=temp_B{k,1};
            end

            lambda.B{k,1} = reshape(temp_B_VB_beta{k,1},D_alpha + d_alpha*num_covariates,r);
            lambda.B{k,1} = tril(lambda.B{k,1});

            m_t_d{k,1}=VB_settings.learning_rate.adapt_tau_1*m_t_d{k,1}+(1-VB_settings.learning_rate.adapt_tau_1)*grad_D{k,1};
            v_t_d{k,1}=VB_settings.learning_rate.adapt_tau_2*v_t_d{k,1}+(1-VB_settings.learning_rate.adapt_tau_2)*(grad_D{k,1}.^2);
            mt_hat_d{k,1}=m_t_d{k,1}./(1-(VB_settings.learning_rate.adapt_tau_1.^t));
            vt_hat_d{k,1}=v_t_d{k,1}./(1-(VB_settings.learning_rate.adapt_tau_2.^t));
            temp_d{k,1} = lambda.d{k,1}+VB_settings.learning_rate.adapt_alpha_d*(mt_hat_d{k,1}./(sqrt(vt_hat_d{k,1})+VB_settings.learning_rate.adapt_epsilon));
            if sum(isnan(temp_d{k,1}))>0
                lambda.d{k,1}=lambda.d{k,1};
            else
                lambda.d{k,1}=temp_d{k,1};
            end
        
        
        
        if VB_settings.store_lambdas == true
            lambda_store{t+1,1} = lambda;
        end
        if (VB_settings.silent == false) && (mod(t,VB_settings.display_period)==0)
            disp(['                 iteration ',num2str(t),'|| smooth LB: ',num2str(round(LB_smooth(t_smooth),1), '%0.1f'),...
                ' standard error: ',num2str(round(std(LBs(~idx))/sum(~idx),2), '%0.2f'),' || running time: ',num2str(round(toc - iter_time,2)),' seconds']);
        end
        % save the output to your directory
        if VB_settings.save == true
            if mod(t,VB_settings.save_period)==0
                if VB_settings.store_lambdas == true
                    save([VB_settings.save_path VB_settings.save_name '.mat'],'lambda','lambda_store','LB_smooth','LB','t','lambda_best','max_best','model','VB_settings');
                else
                    save([VB_settings.save_path VB_settings.save_name '.mat'],'lambda','LB_smooth','LB','t','lambda_best','max_best','model','VB_settings');
                end
            end
        end
        iter_time = toc;
        t = t + 1;
        t_smooth = t_smooth + 1;
        
        if mod(t,1000)==0
           save('LB_store_JS_q.mat','LB','LB_smooth','lambda'); 
        end
        
        
    end
    if converge == "no"
        disp(['    Warning: VB might not converge yet',...
                        '(Initial LB = ',num2str(round(LB(1),1), '%0.1f'),...
                        ' || max LB = ',num2str( round(max_best,1), '%0.1f'),')']);
    elseif converge == "yes"
        disp(['    VB converges to a good local mode',...
                        '(Initial LB = ',num2str(round(LB(1),1), '%0.1f'),...
                        ' || max LB = ',num2str( round(max_best,1), '%0.1f'),')']);
    elseif converge == "reach_max_iter"
        disp(['    VB reaches the maximum number of iterations ',...
                        '(Initial LB = ',num2str(round(LB(1),1), '%0.1f'),...
                        ' || max LB = ',num2str( round(max_best,1), '%0.1f'),')']);
    end

%% Save the output

    VB_results.lambda = lambda_best;
    VB_results.lambda_last = lambda;
    VB_results.LB = LB(1:t-1);
    VB_results.LB_smooth = LB_smooth(1:t_smooth-1);
    VB_results.max_LB = max_best;
    VB_results.converge = converge;   
    VB_results.running_time = toc;    
    
    
VB_results.initial_running_time = initial_running_time;
save([save_path, save_name,'.mat'],'VB_results','VB_settings','model'); 