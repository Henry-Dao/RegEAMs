function y = q_VAFC(theta,mu,B,d)
% INPUT: lambda = (mu,B,d), theta ~ N(mu,covmat) 
%                 with covmat = lambda.B*lambda.B' + diag(lambda.d.^2);
% OUTPUT: y.log = log (q_lambda)
%         y.theta = inv(BB'+D^2)*(Bz + d.*eps) %gradient of q wrt beta
    [p, r] = size(B);
    z = theta - mu;
    
    B1 = d.\B; % (p by r)
    
    [R,xx] = chol(eye(r) + B1'*B1);
    if xx == 0 %------- version 4 ---------------
        log_det = -(sum(log(d.^2)) + sum(log(diag(R).^2)));
        z1 = z./d; % (z tilde) p by 1  
        z2 = B1'*z1; % (z hat) r by 1
        z3 = (R')\z2; % z double hat
        
        y.grad = -(z1./d - (d.\B1)*inv(R)*inv(R')*z2); 
        
        y.log = -0.5*p*log(2*pi) + 0.5*log_det -0.5*(z1'*z1 - z3'*z3); % most efficient (using determinant lemma)
    else %------- version 3 ---------------
        
        term1 = B./(d.^2); 
       
        term2 = B'*term1; % 
        term2(1:r+1:end) = diag(term2) + 1;  
        
        precision_matrix = -term1*(term2\(term1')); % Woodbury: 
        precision_matrix(1:p+1:end) = d.^(-2) + diag(precision_matrix); % Check: sum(abs(precision_matrix - precision_matrixb)<1e-6,'all')
        
        
        y.grad = - precision_matrix*z;
        y.log = -0.5*p*log(2*pi) + 0.5*(-logdet(term2) -sum(log(d.^2))) ...
            +0.5*z'*y.grad; 
    end 
end
