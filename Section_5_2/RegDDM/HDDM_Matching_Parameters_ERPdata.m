function omega_ij = HDDM_Matching_Parameters_ERPdata_ScottHung(model,subject_j,alpha_j_R,beta_vec)

% alpha = ( v0 , v, log(sv), log(a-z-sz/2), log(z-sz/2), log(sz),
%         log(tau^0 - stau/2), log(tau), log(stau) )
    D_alpha = model.num_random_effects;
    n_j = subject_j.num_trials;
    R = size(alpha_j_R,1);
    beta_matrix = reshape(beta_vec,model.beta_dim(1),model.beta_dim(2));
    betaX_j = subject_j.X*beta_matrix';
    u_ij_R =  zeros(R*n_j,D_alpha);

    alpha_v_0 = alpha_j_R(:,1);   alpha_v = alpha_j_R(:,2);
    alpha_tau0 = alpha_j_R(:,7);   alpha_tau = alpha_j_R(:,8);  
     
    same_idx_stack = repmat(subject_j.S,R,1)  == 1; % S = 1 (same)
    same_idx = subject_j.S==1;
 
    Rotation_angles = double(subject_j.E*45 -45);
    v_ij = kron(-alpha_v_0,ones(n_j,1)) - kron(alpha_v,Rotation_angles)   + repmat(betaX_j,R,1);
    v_ij(same_idx_stack) = kron(alpha_v_0,ones(sum(same_idx),1)) + kron(alpha_v,Rotation_angles(same_idx)) + repmat(betaX_j(same_idx,:),R,1);

    tau_ij = kron(exp(alpha_tau0),ones(n_j,1)) + kron(exp(alpha_tau),Rotation_angles);

    idx = [model.u_index{1}, model.u_index{6}];
    u_ij_R(:,idx) = [v_ij log(tau_ij)];

    u_idx = [model.u_index{2}, model.u_index{3}, model.u_index{4},...
            model.u_index{5}, model.u_index{7}];
    model.alpha_index.v0 = 1;   model.alpha_index.v = 2;    model.alpha_index.sv = 3;
    model.alpha_index.azsz = 4;   model.alpha_index.zsz = 5;    model.alpha_index.sz = 6; 
    model.alpha_index.tau0 = 7;   model.alpha_index.tau = 8;     model.alpha_index.stau = 9;

    alpha_idx = [model.alpha_index.sv, model.alpha_index.azsz, model.alpha_index.zsz,...
            model.alpha_index.sz, model.alpha_index.stau];
    u_ij_R(:,u_idx) = kron(alpha_j_R(:,alpha_idx),ones(n_j,1));   

    % ---------- Transform random effects to the natural form ---------
    
    z_ij_tilde = model.T_inv(u_ij_R,model.u_index);
    mu_v_ij = z_ij_tilde(:,model.u_index{1});
    sv_ij = z_ij_tilde(:,model.u_index{2});   
    a_ij = z_ij_tilde(:,model.u_index{3}); 
    z_ij = z_ij_tilde(:,model.u_index{4}); 
    sz_ij = z_ij_tilde(:,model.u_index{5}); 
    mut0_ij = z_ij_tilde(:,model.u_index{6});
    st0_ij = z_ij_tilde(:,model.u_index{7});
        
    omega_ij = [mu_v_ij sv_ij a_ij z_ij sz_ij mut0_ij st0_ij];
    
end
