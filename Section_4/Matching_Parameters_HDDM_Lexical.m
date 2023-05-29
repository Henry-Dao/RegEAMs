function theta_ij = Matching_Parameters_HDDM_Lexical(model,data_subject_j,alpha_j)

%% DESCRIPTION: 
% This function is used to match the subset of random effects to correct observation in Forstmann experiment.
% INPUT: model = structure that contains model specifications
%        alpha = tranformed random effects (column vector/ matrix)
%        data_subject_j = a structure contains all observations from
%        subject j
% OUTPUT: z_ij = the natural random effects that match with the
% osbservations

% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com
%%     
    R = size(alpha_j,1); 
    
    % ---------- Transform random effects to the natural form ---------
    theta_j = model.T_inv(alpha_j);
    v_j = theta_j(1:4);
    sv_j = theta_j(5); % log(sv)
    a_j = theta_j(6:7);
    z_j = theta_j(8:9);
    sz_j = theta_j(10);
    mu_t0_j = theta_j(11);  
    st0_j = theta_j(12);
    

    % ---------- Match the random effects with each observation ---------   
    
    
    n_j = data_subject_j.num_trials;
    
        % ----------- drift rate v -----------------
        W_j = data_subject_j.W;
        I_hf = (W_j == 1); I_lf = (W_j == 2);   I_vlf = (W_j == 3); 
        I_nw = (W_j == 4);
        M = [I_hf I_lf I_vlf I_nw];
        v_ij = reshape(M*v_j',n_j*R,1);  

        % ----------- standard deviation  -----------------
        
        sv_ij = kron(sv_j,ones(n_j,1)); 
        
        % ----------- Boundary Speration a -----------------
        E_j = data_subject_j.E;
        
        I_sp = (E_j == 2); I_acc = (E_j == 1);  
        M = [I_sp I_acc];
        a_ij = reshape(M*a_j',n_j*R,1);  
        z_ij = reshape(M*z_j',n_j*R,1);  
        
        sz_ij = kron(sz_j,ones(n_j,1));
        
        mu_t0_ij = kron(mu_t0_j,ones(n_j,1)); 
        st0_ij = kron(st0_j,ones(n_j,1)); 
        
     
        theta_ij = [v_ij,sv_ij,a_ij,z_ij, sz_ij, mu_t0_ij, st0_ij];
end