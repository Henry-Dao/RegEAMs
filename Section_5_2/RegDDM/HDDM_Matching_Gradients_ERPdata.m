function [grad_alpha_j, grad_beta_matrix_j] = HDDM_Matching_Gradients_ERPdata_ScottHung(model,subject_j,pdf_j,alpha_j_R,beta_vec)

% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com

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

    D_alpha = model.num_random_effects;
    d = model.beta_dim(2);
    d_alpha = model.beta_dim(1);
    n_j = subject_j.num_trials;
    
    R_j = subject_j.R; % (1 = same, 2 = mirror)
    E_j = subject_j.E; % (1 = 0, 2 = 45, 3 = 90, 4 = 135, 5 = 180)
    S_j = subject_j.S; % (1 = same, 2 = mirror)    
%% Gradient wrt alpha_ij   

    grad_mu_v = pdf_j.grad(:,1); 
    grad_sv = pdf_j.grad(:,2);
    grad_a = pdf_j.grad(:,3);
    grad_z = pdf_j.grad(:,4);
    grad_sz = pdf_j.grad(:,5);
    grad_t0 = pdf_j.grad(:,6);
    grad_st0 = pdf_j.grad(:,7);
    grad_u_ij = [grad_mu_v [grad_sv grad_a (grad_a + grad_z) ...
        (grad_a + 0.5*grad_z + grad_sz) grad_t0 (0.5*grad_t0 + grad_st0)]...
        .*exp(u_ij_R(:,model.u_index{2}:model.u_index{7}))];

idx = isfinite(grad_u_ij(:,model.u_index{2})); % Discard observations resulting in NaN or Inf gradients
grad_alpha_j_sv = sum(grad_u_ij(idx,model.u_index{2}),1,'omitnan'); % acutually using isfinite() we no longer need 'omitnan'

idx = isfinite(grad_u_ij(:,model.u_index{3}));
grad_alpha_j_azsz = sum(grad_u_ij(idx,model.u_index{3}),1,'omitnan');

idx = isfinite(grad_u_ij(:,model.u_index{4}));
grad_alpha_j_zsz = sum(grad_u_ij(idx,model.u_index{4}),1,'omitnan');

idx = isfinite(grad_u_ij(:,model.u_index{5}));
grad_alpha_j_sz = sum(grad_u_ij(idx,model.u_index{5}),1,'omitnan');

idx = isfinite(grad_u_ij(:,model.u_index{7}));
grad_alpha_j_stau = sum(grad_u_ij(idx,model.u_index{7}),1,'omitnan');

idx = isfinite(grad_u_ij(:,model.u_index{6}));
grad_u_ij_taustau = grad_u_ij(idx,model.u_index{6});
E_j = subject_j.E; % (1 = 0, 2 = 45, 3 = 90, 4 = 135, 5 = 180)

u_ij_tau = u_ij_R(idx,model.u_index{6}); % u_ij_tau = log(tau_ij - stau_ij/2)

E = E_j*45 -45; E = double(E);
grad_alpha_j_tau0 = sum(grad_u_ij_taustau.*exp(alpha_j_R(model.alpha_index.tau0) - u_ij_tau),1);
grad_alpha_j_tau = sum(grad_u_ij_taustau.*exp(alpha_j_R(model.alpha_index.tau) - u_ij_tau).*E(idx),1);

idx = isfinite(grad_u_ij(:,model.u_index{1}));
grad_u_ij_v = grad_u_ij(:,model.u_index{1});
I_0 = (E_j == 1); I_45 = (E_j == 2);   I_90 = (E_j == 3);
I_135 = (E_j == 4); I_180 = (E_j == 5);
M = double([I_0 I_45 I_90 I_135 I_180]).*[0 45 90 135 180];

S_j = subject_j.S;
grad_alpha_j_v0 = sum(grad_u_ij_v(S_j == 1 & idx),'omitnan') - sum(grad_u_ij_v(S_j == 2 & idx),'omitnan');
grad_alpha_j_v = sum(grad_u_ij_v(S_j == 1 & idx).*M(S_j == 1 & idx,:),'all','omitnan') -... 
    sum(grad_u_ij_v(S_j == 2 & idx).*M(S_j == 2 & idx,:),'all','omitnan');


grad_alpha_j = [ grad_alpha_j_v0, grad_alpha_j_v, grad_alpha_j_sv, grad_alpha_j_azsz,...
    grad_alpha_j_zsz, grad_alpha_j_sz, grad_alpha_j_tau0, grad_alpha_j_tau, grad_alpha_j_stau];

grad_beta_matrix_j = sum(sum(grad_u_ij(:,model.u_index{1}),2).*subject_j.X,1);
end