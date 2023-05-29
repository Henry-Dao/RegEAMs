function z_j = T_inv(alpha_j)

% z_j = [v0_lure v0_target v0_nontarget sv0 a0 z0 sz0 tau0 stau0 v2_lure v2_target v2_nontarget sv2 a2 z2 sz2 tau2 stau2 ] is a row vector

% alpha_j_r = ( log(-v0_lure), log(v0_target), log(-v0_nontarget), log(sv0), 
%               log(a0-z0-sz0/2), log(z0-sz0/2), log(sz0), log(tau0 - stau0/2), log(stau0),
%               log(-v2_lure), log(v2_target), log(-v2_nontarget), log(sv2), 
%               log(a2-z2-sz2/2), log(z2-sz2/2), log(sz2), log(tau2 - stau2/2), log(stau2))

v0_lure = -exp(alpha_j(1));
v0_target = exp(alpha_j(2));
v0_nontarget = -exp(alpha_j(3));
sv0 = exp(alpha_j(4));
sz0 = exp(alpha_j(7));
z0 =  exp(alpha_j(6)) + 0.5*sz0;
a0 = exp(alpha_j(5)) + z0 + 0.5*sz0;
stau0 = exp(alpha_j(9));
tau0 = exp(alpha_j(8)) + 0.5*stau0;

v2_lure = -exp(alpha_j(10));
v2_target = exp(alpha_j(11));
v2_nontarget = -exp(alpha_j(12));
sv2 = exp(alpha_j(13));
sz2 = exp(alpha_j(16));
z2 =  exp(alpha_j(15)) + 0.5*sz2;
a2 = exp(alpha_j(14)) + z2 + 0.5*sz2;
stau2 = exp(alpha_j(18));
tau2 = exp(alpha_j(17)) + 0.5*stau2;

z_j = [v0_lure v0_target v0_nontarget sv0 a0 z0 sz0 tau0 stau0 v2_lure v2_target v2_nontarget sv2 a2 z2 sz2 tau2 stau2 ];
end
