function theta_ij = T_inv_HDDM_Lexical(alpha_ij)

    theta_ij = alpha_ij;
    if(size(theta_ij,2) >1) % if theta_ij is a matrix
        theta_ij(:,5) = exp( alpha_ij(:,5) ); % sv
        theta_ij(:,10) = exp( alpha_ij(:,10) ); % sz
        theta_ij(:,8:9) = exp( alpha_ij(:,8:9) ) + 0.5*exp( alpha_ij(:,10) ); % z
        theta_ij(:,6:7) = exp( alpha_ij(:,6:7) ) + theta_ij(:,8:9) + 0.5*theta_ij(:,10); % a
        theta_ij(:,12) = exp( alpha_ij(:,12) ); % st0
        theta_ij(:,11) = exp(alpha_ij(:,11)) + 0.5*theta_ij(:,12); % t0 
        
    else % theta_ij = theta_j a column/row vector
        theta_ij(5) = exp( alpha_ij(5) ); % sv
        theta_ij(10) = exp( alpha_ij(10) ); % sz
        theta_ij(8:9) = exp( alpha_ij(8:9) ) + 0.5*exp( alpha_ij(10) ); % z
        theta_ij(6:7) = exp( alpha_ij(6:7) ) + theta_ij(8:9) + 0.5*theta_ij(10); % a
        theta_ij(12) = exp( alpha_ij(12) ); % st0
        theta_ij(11) = exp(alpha_ij(11)) +  0.5*theta_ij(12); % t0 
    end
end