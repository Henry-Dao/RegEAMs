function f = DDM_density(model,subject_j,theta_ij,gradient)
% Author: viethung.unsw@gmail.com
%% Inputs:
Lengedre_roots = [ -0.973906528517172 -0.865063366688985 -0.679409568299024 -0.433395394129247 -0.148874338981631 ...
                        0.148874338981631  0.433395394129247  0.679409568299024  0.865063366688985  0.973906528517172]';
Lengedre_weights = [ 0.066671344308688 0.149451349150581 0.219086362515982  0.269266719309996 0.295524224714753 ...
                     0.295524224714753 0.269266719309996 0.219086362515982  0.149451349150581 0.066671344308688]';
    K = -model.kappa:1:model.kappa; % K = (1,2,...,kappa)
        
    weight = model.mixture_weight;
    number_of_nodes = length(Lengedre_roots);
     
    
    
    mu_t0 = theta_ij(:,6); 
    st0 = theta_ij(:,7); 
    
    L_t0 = mu_t0 - 0.5*st0;    U_t0 = mu_t0 + 0.5*st0;
    ind = subject_j.RT > L_t0;
    if sum(ind) == 0
        f.logs = ones(subject_j.num_trials,1)*log((1-weight)*model.p_0);
        f.log = subject_j.num_trials*log((1-weight)*model.p_0);
        f.grad = zeros(subject_j.num_trials,7);
    else        
    
        RE = subject_j.RE(ind);
        RT = subject_j.RT(ind);
        c = RE == "upper";
        n_j = length(RT); 
        
        mu_v = (-c).*theta_ij(ind,1) + (1-c).*theta_ij(ind,1);
        s2_v = theta_ij(ind,2).^2;
        mu_z = theta_ij(ind,4); 
        sz = theta_ij(ind,5); 
        a = theta_ij(ind,3);
        mu_t0 = mu_t0(ind);
        st0 = st0(ind);
        L_t0 = mu_t0 - 0.5*st0;    U_t0 = mu_t0 + 0.5*st0;
        minU_t0 = min(U_t0,RT);
    
    %     Duplicate vectors
        t_vec = kron(RT,ones(number_of_nodes^2,1) );
        c_vec = kron(c,ones(number_of_nodes^2,1) ); % c = 1 (upper); c = 0 (lower); 
        
        mu_t0_vec = kron(mu_t0,ones(number_of_nodes^2,1) );
        st0_vec = kron(st0,ones(number_of_nodes^2,1) );
        
        minU_t0_vec = kron(minU_t0,ones(number_of_nodes^2,1) );
        U_t0_vec = kron(U_t0,ones(number_of_nodes^2,1) );
        L_t0_vec = kron(L_t0,ones(number_of_nodes^2,1) );
        
        Xi_t = repmat(kron(Lengedre_roots,ones(number_of_nodes,1)),n_j,1);
        t_0_vec = 0.5*(minU_t0_vec - L_t0_vec).*Xi_t + 0.5*(minU_t0_vec+L_t0_vec);
        
        mu_z_vec = kron(mu_z,ones(number_of_nodes^2,1) ); 
        sz_vec = kron(sz,ones(number_of_nodes^2,1) ); 
        Xi_z = repmat(Lengedre_roots,number_of_nodes*n_j,1); 
        z_i = 0.5*sz_vec.*Xi_z + mu_z_vec;    
        
        W_t_z = repmat(Lengedre_weights,number_of_nodes*n_j,1).*repmat(kron(Lengedre_weights,ones(number_of_nodes,1)),n_j,1);
        a_vec = kron(a,ones(number_of_nodes^2,1) ); 
        z_i = c_vec.*(a_vec-z_i) + (1-c_vec).*z_i;
        t_d_vec = t_vec-t_0_vec;    
        
        s2_v_vec = kron(s2_v,ones(number_of_nodes^2,1) );
        mu_v_vec = kron(mu_v,ones(number_of_nodes^2,1) );
        
        A = 1 + t_d_vec.*s2_v_vec;   B = mu_v_vec - z_i.*s2_v_vec;     C = (z_i + 2*K.*a_vec).^2./t_d_vec;% shortcut terms
        
        smallest_term = min(C,[],'all');
        exp_part = exp( 0.5*smallest_term -0.5*C - 0.5*mu_v_vec.^2./s2_v_vec + 0.5*B.^2./(A.*s2_v_vec ) );
        Normalized = 0.25*(minU_t0_vec - L_t0_vec)./(U_t0_vec - L_t0_vec);
        W_k_scaled = (2*K + z_i./a_vec).*exp_part;
        sum_W_k_scaled = sum(W_k_scaled,2); 
        if sum(isinf(sum_W_k_scaled))>0 || sum(isnan(sum_W_k_scaled))>0
            smallest_term = 0;
            exp_part = exp( 0.5*smallest_term -0.5*C - 0.5*mu_v_vec.^2./s2_v_vec + 0.5*B.^2./(A.*s2_v_vec ) );
            Normalized = 0.25*(minU_t0_vec - L_t0_vec)./(U_t0_vec - L_t0_vec);
            W_k_scaled = (2*K + z_i./a_vec).*exp_part;
            sum_W_k_scaled = sum(W_k_scaled,2);
        end
        g_density_scaled = 1./sqrt(A).*(t_d_vec.^(-1.5)).*a_vec/sqrt(2*pi).*sum_W_k_scaled; % compared with above formula
        temp_scaled = Normalized.*(W_t_z.*g_density_scaled);
        
        
        pdf = weight*sum(reshape(temp_scaled,number_of_nodes^2,n_j))'/exp(0.5*smallest_term) + (1-weight)*model.p_0; 
        f.logs = zeros(subject_j.num_trials,1);
        f.logs(ind) = log(pdf);
        f.logs(~ind) = log((1-weight)*model.p_0);
        f.log = sum(f.logs,'omitnan');
        if gradient == true
            % ------------- Partial derivatives  of g_density wrt natural parameters (mu_v,s2_v, a, z, tau) ----------------
            grad_mu_v_scaled =   ((-1).^c_vec).*temp_scaled.*(-(t_d_vec.*mu_v_vec + z_i)./A );
            
            grad_s_v_scaled =  2*sqrt(s2_v_vec).* temp_scaled.*( -0.5*t_d_vec./A + ( mu_v_vec.^2.*A.^2 - 2*z_i.*B.*A.*s2_v_vec - (1 + 2*t_d_vec.*s2_v_vec).*B.^2 )./(2*A.^2.*s2_v_vec.^2) ) ; % scalar
            
            grad_z_i_term1_scaled = g_density_scaled.*( -B./A );
            grad_z_i_term2_scaled = (1./sqrt(A).*(t_d_vec.^(-1.5)).*a_vec/sqrt(2*pi)).*sum(exp_part.*(1./a_vec - C./a_vec),2);
            grad_z_i_scaled = (-1).^c_vec.*Normalized.*(W_t_z.*(grad_z_i_term1_scaled + grad_z_i_term2_scaled)); % the same as above. Here use W_k.
            
            
            grad_a_i_term1_scaled =  g_density_scaled./a_vec;
            grad_a_i_term2_scaled = -(1./sqrt(A).*(t_d_vec.^(-1.5)).*a_vec/sqrt(2*pi)).*sum(exp_part.*(z_i./a_vec.^2 + 2*K.*C./a_vec),2);
            
            grad_a_i_scaled =  Normalized.*(W_t_z.*(grad_a_i_term1_scaled + grad_a_i_term2_scaled));
            grad_a_scaled = grad_a_i_scaled - c_vec.*grad_z_i_scaled; % calar
            
            term1_scaled = -0.5*(s2_v_vec./A);
            term2_scaled  = -0.5*B.^2./A.^2;
            term3_scaled  = -1.5./t_d_vec;
            term4_scaled  = 1./sqrt(A).*(t_d_vec.^(-1.5)).*a_vec/sqrt(2*pi).*sum(((2*a_vec.*K + z_i).^3./(2*a_vec.*t_d_vec.^2)).*exp_part,2);
            grad_t_d_scaled = Normalized.*(W_t_z.*(g_density_scaled.*(term1_scaled + term2_scaled + term3_scaled) + term4_scaled));
            grad_t0_j_scaled = - grad_t_d_scaled; % this is a vector
            
            % ------------- Partial derivatives  of g_density wrt transformed parameters theta ----------------
            
            grad_mu_z_scaled = grad_z_i_scaled;
            grad_sz_scaled = grad_z_i_scaled.*Xi_z/2;
            
            
            grad_mu_t0_scaled = (t_vec<U_t0_vec).*(-temp_scaled.*(U_t0_vec - L_t0_vec)./(st0_vec.*(minU_t0_vec - L_t0_vec)) + grad_t0_j_scaled.*(1 - Xi_t)/2) +...
                (t_vec>U_t0_vec).*grad_t0_j_scaled;
            grad_s_t_scaled =  (t_vec<U_t0_vec).*(-temp_scaled.*(U_t0_vec - L_t0_vec).*(t_vec-mu_t0_vec)./(st0_vec.^2.*(minU_t0_vec - L_t0_vec)) + grad_t0_j_scaled.*(Xi_t-1)/4) +...
                (t_vec>U_t0_vec).*grad_t0_j_scaled.*Xi_t/2;
            
            % ------------- Partial derivatives  of g_density wrt transformed parameters theta ----------------
                        
            grad_mu_v = sum(reshape(grad_mu_v_scaled,number_of_nodes^2,n_j))'; % column vector
            grad_sv = sum(reshape(grad_s_v_scaled,number_of_nodes^2,n_j))'; % column vector
            grad_a = sum(reshape(grad_a_scaled,number_of_nodes^2,n_j))'; % derivative of p_hat(i) wrt theta3
            
            grad_z = sum(reshape(grad_mu_z_scaled,number_of_nodes^2,n_j))'; % derivative of p_hat(i) wrt theta4
            grad_sz = sum(reshape(grad_sz_scaled,number_of_nodes^2,n_j))'; % derivative of p_hat(i) wrt theta5
            
            grad_t0 = sum(reshape(grad_mu_t0_scaled,number_of_nodes^2,n_j))'; % derivative of g_density wrt theta6
            grad_st0 = sum(reshape(grad_s_t_scaled,number_of_nodes^2,n_j))'; % derivative of g_density wrt theta7
            
            
            Gradients = [grad_mu_v grad_sv grad_a  grad_z grad_sz grad_t0 grad_st0]/exp(0.5*smallest_term);
            
            f.grad = zeros(subject_j.num_trials,7);
            f.grad(ind,:) = weight*Gradients./pdf;
        end
    end
end