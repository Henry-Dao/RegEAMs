function output = Hybrid_VAFC_v6(model,data,VB_settings)
% INPUT: model = structure that contains model specifications
%        data = the data (in the correct format, see the user manual)
%        likelihood = likelihood function
%        prior_density = prior density
%        VB_setting = Settings of VB
%             initial = intial value for lambda( structure with 3 fields: lambda.mu, lambda.B and lambda.d)
%             r = number of factors used to parameterize the covariance matrix of q()
%             I = number of MC samples used to estimate the lower bound and the gradients
%             max_iter = the maximum number of iterations
%             window = window size to compute the average of the lower bounds
%             patience_parameter = the number of consecutive iterations in which the LB does not increase
%             max_norm = gradient clipping
%             learning_rate = ADADELTA learning rate parameter (v & eps)
%             silent = "no" (print out iteration t and the estimated lower bounds LB)

% OUTPUT: is a structure that contains several fields
%     output.lambda = the best lambda (corresponds to the highest lower bound);
%     output.LB = all the estimated lower bounds
%     output.LB_smooth = all the averaged lower bounds
%     output.max_LB = the highest lower bound
%     output.converge = "true" if the optimization converges and "false" if it fails to converge; 

% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com

%% Initial Stage: run the first window iterations
tic
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

    p = D_alpha*(J + 2);

    v_a = model.prior_par.v_a;
    df = v_a + 2*D_alpha + J + 2;
    
    if VB_settings.learning_rate == "ADADELTA"
        v = VB_settings.ADADELTA.v; eps = VB_settings.ADADELTA.eps; 
        E_g2 = VB_settings.ADADELTA.E_g2; E_delta2 = VB_settings.ADADELTA.E_delta2;
    elseif VB_settings.learning_rate == "ADAM"
        m_t_mu = VB_settings.ADAM.m_t_mu;
        v_t_mu = VB_settings.ADAM.v_t_mu;

        m_t_B = VB_settings.ADAM.m_t_B;
        v_t_B = VB_settings.ADAM.v_t_B;

        m_t_d = VB_settings.ADAM.m_t_d;
        v_t_d = VB_settings.ADAM.v_t_d;
    end
iter_time = toc; 
if ~isfield(VB_settings,'LB')
    for t = 1:window
        mu = lambda.mu;
        B = lambda.B;
        d = lambda.d;
        parfor i = 1:I
        % (1) Generate theta  
            epsilon = randn(p,1);
            z = randn(r,1);
            theta_1 = mu + B*z + d.*epsilon; % theta_1 = (alpha_1,...,alpha_J,mu_alpha,log a_1,...,log a_D)
            ALPHA = reshape(theta_1(1:D_alpha*J),D_alpha,J);
            mu_alpha = theta_1(D_alpha*J + 1:D_alpha*(J+1));
            log_a = theta_1(D_alpha*(J+1)+1:end);
            a = exp(log_a);
            
            Psi = 2*v_a*diag(1./a);
            for j=1:J
                Psi = Psi + (ALPHA(:,j)-mu_alpha)*(ALPHA(:,j)-mu_alpha)';
            end            
            Sigma_alpha = iwishrnd(Psi,df);
        % (2) Calculate the likelihood, prior and q_vb    
            like = Likelihood(model,data,ALPHA); % last argument = "true" means to compute the gradients 
            prior = prior_density(model,ALPHA,mu_alpha,Sigma_alpha,a); 
            q_lambda = q_vb(theta_1,mu,B,d); 
            q_Sigma = p_Sigma_alpha(Sigma_alpha,df,v_a,Psi,ALPHA,mu_alpha,a);

        % (3) Estimate the lower bound
            LBs(i) = like.log +  prior.log - q_lambda.log - q_Sigma.log;
        % (4) Cumpute the gradients
            grad_theta_1_LB = like.grad + prior.grad - q_lambda.grad - q_Sigma.grad;
            temp = grad_theta_1_LB;
            gradmu(:,i) = temp;
            gradB(:,:,i) = temp*z';
            gradd(:,i) = temp.*epsilon;   
        end
        
        % Estimate the gradients
            grad_mu = mean(gradmu,2);
            grad_B = mean(gradB,3);
            grad_D = mean(gradd,2);

            g = [grad_mu;grad_B(:);grad_D]; % Stack gradient of LB into 1 column 
        % Gradient clipping
            norm_g = norm(g);
            if norm_g > max_norm
                g = max_norm*g/norm_g;
            end
        % Update learning rate and Lambda
            if VB_settings.learning_rate == "ADADELTA"
                E_g2 = v*E_g2 + (1-v)*g.^2;
                rho = sqrt(E_delta2 + eps)./sqrt(E_g2+eps);
                Delta = rho.*g;
                E_delta2 = v*E_delta2 + (1-v)*Delta.^2;
                vec_lambda = [lambda.mu;lambda.B(:);lambda.d];
                vec_lambda = vec_lambda + Delta;
                lambda.mu = vec_lambda(1:p);
                %             lambda.B = vec2mat(vec_lambda((p+1):(p*(r+1))),p)';
                lambda.B = reshape(vec_lambda((p+1):(p*(r+1))),p,r);
                lambda.B = tril(lambda.B);
                lambda.d = vec_lambda((p*(r+1)+1):end);
            elseif VB_settings.learning_rate == "ADAM"
                m_t_mu=VB_settings.ADAM.adapt_tau_1*m_t_mu+(1-VB_settings.ADAM.adapt_tau_1)*grad_mu;
                v_t_mu=VB_settings.ADAM.adapt_tau_2*v_t_mu+(1-VB_settings.ADAM.adapt_tau_2)*(grad_mu.^2);
                mt_hat_mu=m_t_mu./(1-(VB_settings.ADAM.adapt_tau_1.^t));
                vt_hat_mu=v_t_mu./(1-(VB_settings.ADAM.adapt_tau_2.^t));
                temp_mu = lambda.mu+VB_settings.ADAM.adapt_alpha_mu*(mt_hat_mu./(sqrt(vt_hat_mu)+VB_settings.ADAM.adapt_epsilon));
                if sum(isnan(temp_mu))>0
                    lambda.mu=lambda.mu;
                else
                    lambda.mu=temp_mu;
                end
    
                m_t_B=VB_settings.ADAM.adapt_tau_1*m_t_B+(1-VB_settings.ADAM.adapt_tau_1)*grad_B(:);
                v_t_B=VB_settings.ADAM.adapt_tau_2*v_t_B+(1-VB_settings.ADAM.adapt_tau_2)*(grad_B(:).^2);
                mt_hat_B=m_t_B./(1-(VB_settings.ADAM.adapt_tau_1.^t));
                vt_hat_B=v_t_B./(1-(VB_settings.ADAM.adapt_tau_2.^t));
                temp_B = lambda.B(:) + VB_settings.ADAM.adapt_alpha_B*(mt_hat_B./(sqrt(vt_hat_B)+VB_settings.ADAM.adapt_epsilon));
                if sum(isnan(temp_B))>0
                    temp_B_VB_beta=lambda.B(:);
                else
                    temp_B_VB_beta=temp_B;
                end
    
                lambda.B = reshape(temp_B_VB_beta,p,r);
                lambda.B = tril(lambda.B);
    
                m_t_d=VB_settings.ADAM.adapt_tau_1*m_t_d+(1-VB_settings.ADAM.adapt_tau_1)*grad_D;
                v_t_d=VB_settings.ADAM.adapt_tau_2*v_t_d+(1-VB_settings.ADAM.adapt_tau_2)*(grad_D.^2);
                mt_hat_d=m_t_d./(1-(VB_settings.ADAM.adapt_tau_1.^t));
                vt_hat_d=v_t_d./(1-(VB_settings.ADAM.adapt_tau_2.^t));
                temp_d = lambda.d+VB_settings.ADAM.adapt_alpha_d*(mt_hat_d./(sqrt(vt_hat_d)+VB_settings.ADAM.adapt_epsilon));
                if sum(isnan(temp_d))>0
                    lambda.d=lambda.d;
                else
                    lambda.d=temp_d;
                end
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
else 
    LB(1:window) = VB_settings.LB(end-window+1:end);
    t = window;
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
        
        parfor i = 1:I
        % (1) Generate theta  
            epsilon = randn(p,1);
            z = randn(r,1);
            theta_1 = mu + B*z + d.*epsilon; % theta_1 = (alpha_1,...,alpha_J,mu_alpha,log a_1,...,log a_D)
            ALPHA = reshape(theta_1(1:D_alpha*J),D_alpha,J);
            mu_alpha = theta_1(D_alpha*J + 1:D_alpha*(J+1));
            log_a = theta_1(D_alpha*(J+1)+1:end);
            a = exp(log_a);
            
            Psi = 2*v_a*diag(1./a);
            for j=1:J
                Psi = Psi + (ALPHA(:,j)-mu_alpha)*(ALPHA(:,j)-mu_alpha)';
            end            
            Sigma_alpha = iwishrnd(Psi,df);
        % (2) Calculate the likelihood, prior and q_vb          
            like = Likelihood(model,data,ALPHA); % last argument = "true" means to compute the gradients 
            prior = prior_density(model,ALPHA,mu_alpha,Sigma_alpha,a); 
            q_lambda = q_vb(theta_1,mu,B,d); 
            q_Sigma = p_Sigma_alpha(Sigma_alpha,df,v_a,Psi,ALPHA,mu_alpha,a);

        % (3) Estimate the lower bound
            LBs(i) = like.log +  prior.log - q_lambda.log - q_Sigma.log;

        % (4) Cumpute the gradients
            grad_theta_1_LB = like.grad + prior.grad - q_lambda.grad - q_Sigma.grad;
            temp = grad_theta_1_LB;
            gradmu(:,i) = temp;
            gradB(:,:,i) = temp*z';
            gradd(:,i) = temp.*epsilon;   
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
        grad_mu = mean(gradmu(:,~idx),2);
        grad_B = mean(gradB(:,:,~idx),3);
        grad_D = mean(gradd(:,~idx),2);
        
        g = [grad_mu;grad_B(:);grad_D]; % Stack gradient of LB into 1 column
        % Gradient clipping
        norm_g = norm(g);
        if norm_g > max_norm
            g = max_norm*g/norm_g;
        end
        
        % Update learning rate and Lambda
        if VB_settings.learning_rate == "ADADELTA"
            E_g2 = v*E_g2 + (1-v)*g.^2;
            rho = sqrt(E_delta2 + eps)./sqrt(E_g2+eps);
            Delta = rho.*g;
            E_delta2 = v*E_delta2 + (1-v)*Delta.^2;
            vec_lambda = [lambda.mu;lambda.B(:);lambda.d];
            vec_lambda = vec_lambda + Delta;
            lambda.mu = vec_lambda(1:p);
            %             lambda.B = vec2mat(vec_lambda((p+1):(p*(r+1))),p)';
            lambda.B = reshape(vec_lambda((p+1):(p*(r+1))),p,r);
            lambda.B = tril(lambda.B);
            lambda.d = vec_lambda((p*(r+1)+1):end);
        elseif VB_settings.learning_rate == "ADAM"
            m_t_mu=VB_settings.ADAM.adapt_tau_1*m_t_mu+(1-VB_settings.ADAM.adapt_tau_1)*grad_mu;
            v_t_mu=VB_settings.ADAM.adapt_tau_2*v_t_mu+(1-VB_settings.ADAM.adapt_tau_2)*(grad_mu.^2);
            mt_hat_mu=m_t_mu./(1-(VB_settings.ADAM.adapt_tau_1.^t));
            vt_hat_mu=v_t_mu./(1-(VB_settings.ADAM.adapt_tau_2.^t));
            temp_mu = lambda.mu+VB_settings.ADAM.adapt_alpha_mu*(mt_hat_mu./(sqrt(vt_hat_mu)+VB_settings.ADAM.adapt_epsilon));
            if sum(isnan(temp_mu))>0
                lambda.mu=lambda.mu;
            else
                lambda.mu=temp_mu;
            end

            m_t_B=VB_settings.ADAM.adapt_tau_1*m_t_B+(1-VB_settings.ADAM.adapt_tau_1)*grad_B(:);
            v_t_B=VB_settings.ADAM.adapt_tau_2*v_t_B+(1-VB_settings.ADAM.adapt_tau_2)*(grad_B(:).^2);
            mt_hat_B=m_t_B./(1-(VB_settings.ADAM.adapt_tau_1.^t));
            vt_hat_B=v_t_B./(1-(VB_settings.ADAM.adapt_tau_2.^t));
            temp_B = lambda.B(:) + VB_settings.ADAM.adapt_alpha_B*(mt_hat_B./(sqrt(vt_hat_B)+VB_settings.ADAM.adapt_epsilon));
            if sum(isnan(temp_B))>0
                temp_B_VB_beta=lambda.B(:);
            else
                temp_B_VB_beta=temp_B;
            end

            lambda.B = reshape(temp_B_VB_beta,p,r);
            lambda.B = tril(lambda.B);

            m_t_d=VB_settings.ADAM.adapt_tau_1*m_t_d+(1-VB_settings.ADAM.adapt_tau_1)*grad_D;
            v_t_d=VB_settings.ADAM.adapt_tau_2*v_t_d+(1-VB_settings.ADAM.adapt_tau_2)*(grad_D.^2);
            mt_hat_d=m_t_d./(1-(VB_settings.ADAM.adapt_tau_1.^t));
            vt_hat_d=v_t_d./(1-(VB_settings.ADAM.adapt_tau_2.^t));
            temp_d = lambda.d+VB_settings.ADAM.adapt_alpha_d*(mt_hat_d./(sqrt(vt_hat_d)+VB_settings.ADAM.adapt_epsilon));
            if sum(isnan(temp_d))>0
                lambda.d=lambda.d;
            else
                lambda.d=temp_d;
            end
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
                if VB_settings.learning_rate == "ADADELTA"
                    VB_settings.ADADELTA.E_g2 = E_g2;  
                    VB_settings.ADADELTA.E_delta2 = E_delta2; 
                elseif VB_settings.learning_rate == "ADAM"
                    VB_settings.ADAM.m_t_mu = m_t_mu;
                    VB_settings.ADAM.v_t_mu = v_t_mu;
                    VB_settings.ADAM.m_t_B = m_t_B;
                    VB_settings.ADAM.v_t_B = v_t_B;
                    VB_settings.ADAM.m_t_d = m_t_d;
                    VB_settings.ADAM.v_t_d = v_t_d;
                end
                VB_settings.LB = LB(1:t);
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
%% Draws from VB distribution
if VB_settings.generated_samples == true
    mu = lambda_best.mu; B = lambda_best.B;   d = lambda_best.d;   
    N = 1000;
    ALPHA = zeros(D_alpha,J,N);
    mu_alpha = zeros(D_alpha,N);
    vech_Sigma_alpha = zeros(D_alpha*(D_alpha+1)/2,N);
    vech_C_alpha_star = zeros(D_alpha*(D_alpha+1)/2,N);
    log_a = zeros(D_alpha,N);
    
    for i = 1:N
        epsilon = randn(p,1);
        z = randn(r,1);
        theta_1 = mu + B*z + d.*epsilon; % theta_1 = (alpha_1,...,alpha_J,mu_alpha,log a_1,...,log a_D)
        ALPHA(:,:,i) = reshape(theta_1(1:D_alpha*J),D_alpha,J);
        mu_alpha(:,i) = theta_1(D_alpha*J + 1:D_alpha*(J+1));
        log_a(:,i) = theta_1(D_alpha*(J+1)+1:end);
        a = exp(log_a(:,i));   
        Psi = 2*v_a*diag(1./a);
        for j=1:J
            Psi = Psi + (ALPHA(:,j,i)-mu_alpha(:,i))*(ALPHA(:,j,i)-mu_alpha(:,i))';
        end            
        Sigma_alpha = iwishrnd(Psi,df);
        vech_Sigma_alpha(:,i) = vech(Sigma_alpha);
        C = chol(Sigma_alpha,'lower');
        C_star = C; C_star(1:D_alpha+1:end) = log(diag(C));
        vech_C_alpha_star(:,i) = vech(C_star);     
    end 
    output.theta_VB.alpha = ALPHA;
    output.theta_VB.mu_alpha = mu_alpha;
    output.theta_VB.vech_Sigma_alpha = vech_Sigma_alpha;
    output.theta_VB.vech_C_alpha_star = vech_C_alpha_star;
    output.theta_VB.log_a = log_a;
end
    
%% Save the output
    output.lambda = lambda_best;
    output.lambda_last = lambda;
    output.LB = LB(1:t-1);
    output.LB_smooth = LB_smooth(1:t_smooth-1);
    output.max_LB = max_best;
    output.converge = converge;    
    output.running_time = toc;
end
