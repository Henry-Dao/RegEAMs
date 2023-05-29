function y = q_VAFC_separate(ALPHA,beta_vec_mu_alpha,mu,B,d,J)
% INPUT: lambda = (mu,B,d), theta ~ N(mu,covmat) 
%                 with covmat = lambda.B*lambda.B' + diag(lambda.d.^2);
% OUTPUT: y.log = log (q_lambda)
%         y.theta = inv(BB'+D^2)*(Bz + d.*eps) % gradient of q wrt beta
    combine = ALPHA;
    for k=1:J
        [p, r] = size(B{k,1});
        z{k,1} = combine(:,k) - mu{k,1};

        B1{k,1} = d{k,1}.\B{k,1}; 
        [R{k,1},xx{k,1}] = chol(eye(r) + B1{k,1}'*B1{k,1});
        if xx{k,1} == 0 %------- version 4 ---------------
            log_det{k,1} = -(sum(log(d{k,1}.^2)) + sum(log(diag(R{k,1}).^2)));
            z1{k,1} = z{k,1}./d{k,1};  
            z2{k,1} = B1{k,1}'*z1{k,1}; 
            z3{k,1} = (R{k,1}')\z2{k,1}; 

            temp_grad{k,1} = -(z1{k,1}./d{k,1} - (d{k,1}.\B1{k,1})*inv(R{k,1})*inv(R{k,1}')*z2{k,1}); 
            temp_log(k,1) = -0.5*p*log(2*pi) + 0.5*log_det{k,1} -0.5*(z1{k,1}'*z1{k,1} - z3{k,1}'*z3{k,1}); 
        else %------- version 3 ---------------

            term1{k,1} = B{k,1}./(d{k,1}.^2); 

            term2{k,1} = B{k,1}'*term1{k,1}; 
            term2{k,1}(1:r+1:end) = diag(term2{k,1}) + 1; 
            precision_matrix{k,1} = -term1{k,1}*(term2{k,1}\(term1{k,1}'));
            precision_matrix{k,1}(1:p+1:end) = d{k,1}.^(-2) + diag(precision_matrix{k,1}); 

            temp_grad{k,1} = - precision_matrix{k,1}*z{k,1};
            temp_log(k,1) = -0.5*p*log(2*pi) + 0.5*(-logdet(term2{k,1}) -sum(log(d{k,1}.^2))) ...
                +0.5*z{k,1}'*temp_grad{k,1}; 
        end
    end
    combine = beta_vec_mu_alpha;
    k=J+1;
    [p, r] = size(B{k,1});
    z{k,1} = combine - mu{k,1};

    B1{k,1} = d{k,1}.\B{k,1};
    [R{k,1},xx{k,1}] = chol(eye(r) + B1{k,1}'*B1{k,1});
    if xx{k,1} == 0 %------- version 4 ---------------
        log_det{k,1} = -(sum(log(d{k,1}.^2)) + sum(log(diag(R{k,1}).^2)));
        z1{k,1} = z{k,1}./d{k,1};
        z2{k,1} = B1{k,1}'*z1{k,1};
        z3{k,1} = (R{k,1}')\z2{k,1};

        temp_grad{k,1} = -(z1{k,1}./d{k,1} - (d{k,1}.\B1{k,1})*inv(R{k,1})*inv(R{k,1}')*z2{k,1});
        temp_log(k,1) = -0.5*p*log(2*pi) + 0.5*log_det{k,1} -0.5*(z1{k,1}'*z1{k,1} - z3{k,1}'*z3{k,1});
    else %------- version 3 ---------------

        term1{k,1} = B{k,1}./(d{k,1}.^2);

        term2{k,1} = B{k,1}'*term1{k,1};
        term2{k,1}(1:r+1:end) = diag(term2{k,1}) + 1;
        precision_matrix{k,1} = -term1{k,1}*(term2{k,1}\(term1{k,1}'));
        precision_matrix{k,1}(1:p+1:end) = d{k,1}.^(-2) + diag(precision_matrix{k,1});

        temp_grad{k,1} = - precision_matrix{k,1}*z{k,1};
        temp_log(k,1) = -0.5*p*log(2*pi) + 0.5*(-logdet(term2{k,1}) -sum(log(d{k,1}.^2))) ...
            +0.5*z{k,1}'*temp_grad{k,1};
    end
    y.grad = temp_grad;
    y.log= sum(temp_log);
end
