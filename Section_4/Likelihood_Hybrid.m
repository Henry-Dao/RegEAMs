function output = Likelihood_Hybrid_v5(model,data,alpha)
% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com
%%
    loga = 0; % log of the total likelihood

    D_alpha = model.subject_param_dim; % n_alpha = D_alpha = dimension of random effect alpha
    J = length(data); % number of subjects/participants
    grad_alpha = zeros(D_alpha,J); % store all the gradients wrt alpha_j
    grad_mu = zeros(D_alpha,1);
    grad_loga = zeros(D_alpha,1);

    for j = 1:J  
    %% Match each observation to the correct set of parameters (b,A,v,s,tau)
        omega_ij = model.matching_parameters(model,data{j,1},alpha(:,j)');
        pdf_j = model.density(model,data{j,1},omega_ij,true);
        loga = loga + pdf_j.log;
        
    %% Compute the gradients 
        grad_alpha(:,j) = model.matching_gradients(model,data{j,1},pdf_j,alpha(:,j)');

    end   
%% output of the function
    output.log = loga;
    output.grad = [grad_alpha(:); grad_mu; grad_loga];
end