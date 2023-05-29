function output = Likelihood_Hybrid_informativeprior_Sigma_v5(model,data,alpha,beta_vec)

% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com
%%
    loga = 0; % log of the total likelihood

    D_alpha = model.subject_param_dim; % n_alpha = D_alpha = dimension of random effect alpha
    d = model.beta_dim(2);
    d_alpha = model.beta_dim(1);
    J = length(data); % number of subjects/participants
    grad_alpha = zeros(D_alpha,J); % store all the gradients wrt alpha_j
%     grad_u = zeros(D_alpha,J); % store all the gradients wrt u_j
    grad_beta_matrix = zeros(d_alpha,d);
    grad_mu = zeros(D_alpha,1);
%     grad_loga = zeros(D_alpha,1);
%     beta_matrix = reshape(beta_vec,d_alpha,d);
    for j = 1:J  
    %% Match each observation to the correct set of parameters (b,A,v,s,tau)
%         n_j = data{j,1}.num_trials;
%         u_ij = model.structural_equation(model,data{j,1},alpha(:,j)',beta_vec);
%         omega_ij = model.matching_function_1(model,data{j,1},u_ij);    
        omega_ij = model.matching_parameters(model,data{j,1},alpha(:,j)',beta_vec);
        pdf_j = model.density(model,data{j,1},omega_ij,true);
        loga = loga + pdf_j.log;
        
    %% Compute the gradients 

%         grad_u_ij = model.matching_function_2(model,data{j,1},pdf_j,u_ij);            
%         [grad_alpha(:,j), grad_beta_j_matrix] = model.structural_equation_gradients_matching(model,data{j,1},alpha(:,j),u_ij,grad_u_ij);
        [grad_alpha(:,j), grad_beta_j_matrix] = model.matching_gradients(model,data{j,1},pdf_j,alpha(:,j)',beta_vec);
%         grad_alpha(:,j) = grad_alpha_j;
        grad_beta_matrix = grad_beta_matrix + grad_beta_j_matrix;

    end   
%% output of the function
    output.log = loga;
    output.grad = [grad_alpha(:); grad_beta_matrix(:); grad_mu];
end