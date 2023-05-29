function theta_ij = T_inv_HDDM(alpha_ij,idx)
theta_ij = alpha_ij;
%     theta = (v,sv,a,z,sz,t0,st0)
% alpha = (v, log(sv), log(a-z-sz/2), log(z-sz/2), log(sz), log(t0-st0/2), log(st0) )
if(size(theta_ij,2) >1) % if theta_ij is a matrix
    theta_ij(:,idx{2}) = exp( alpha_ij(:,idx{2}) ); % sv
    theta_ij(:,idx{5}) = exp( alpha_ij(:,idx{5}) ); % sz
    theta_ij(:,idx{4}) = exp( alpha_ij(:,idx{4}) ) + 0.5*theta_ij(:,idx{5}); % z
    theta_ij(:,idx{3}) = exp( alpha_ij(:,idx{3}) ) + theta_ij(:,idx{4}) + 0.5*theta_ij(:,idx{5}); % log(a - z - sz/2)
    theta_ij(:,idx{7}) = exp( alpha_ij(:,idx{7}) ); % log(st0)
    theta_ij(:,idx{6}) = exp(alpha_ij(:,idx{6})) + 0.5*theta_ij(:,idx{7}); % log(t0 -st0/2)
    
else % theta_ij = theta_j a column/row vector
    theta_ij(idx{2}) = exp( alpha_ij(idx{2}) ); % log(sv)
    theta_ij(idx{5}) = exp( alpha_ij(idx{5}) ); % log(sz)
    theta_ij(idx{4}) = exp( alpha_ij(idx{4}) ) + 0.5*theta_ij(idx{5}); % log(z-sz/2)
    theta_ij(idx{3}) = exp( alpha_ij(idx{3}) ) + theta_ij(idx{4}) + 0.5*theta_ij(idx{5}); % log(a - z - sz/2)
    theta_ij(idx{7}) = exp( alpha_ij(idx{7}) ); % log(st0)
    theta_ij(idx{6}) = exp(alpha_ij(idx{6})) +  0.5*theta_ij(idx{7}); % log(t0 -st0/2)
end
end
%% Test: copy and run the code below in a new script file
% idx = {1,2,3,4,5,6,7};
% v = 1.5;    sv = 1;
% a = 1.5;  
% z = 0.7;  sz = 0.6;    
% t0 = 0.35;   st0 = 0.2;
% theta = [v, sv, a, z, sz, t0 ,st0];
% alpha = [v, log([sv, a-z-0.5*sz, z-0.5*sz, sz, t0, t0 - 0.5*st0]);
% T_inv_HDDM(alpha,idx)
% 
% % theta_vector = repmat(theta,3,1);
% % alpha_vector = repmat(alpha,3,1);
% % T_inv_HDDM(alpha_vector,idx)