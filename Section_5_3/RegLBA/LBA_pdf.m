function output  = LBA_pdf(model,subject_j,z_ij,gradient)

% OUTPUT: .log = log(LBA(c,t));
%         .grad_b = gradient of  log(LBA(c,t)) wrt b

% Author: Viet-Hung Dao (UNSW)
% Email: viethung.unsw@gmail.com

    b = z_ij(:,1:2);    A = z_ij(:,3:4);
    v = z_ij(:,5:6);    s = z_ij(:,7:8);    tau = z_ij(:,9:10);

idx = (subject_j.RT > tau(:,1)) & (subject_j.RT > tau(:,2));
if sum(idx) == 0
    output.logs = ones(subject_j.num_trials,1)*log((1-model.mixture_weight)*model.p_0);
    output.log = subject_j.num_trials*log((1-model.mixture_weight)*model.p_0);
    output.grad = zeros(subject_j.num_trials,10);
else

    f_c = pdf_c(subject_j.RT(idx),b(idx,1),A(idx,1),v(idx,1),s(idx,1),tau(idx,1),gradient);
    
    F_k = CDF_c(subject_j.RT(idx),b(idx,2),A(idx,2),v(idx,2),s(idx,2),tau(idx,2),gradient);

    mixture_pdf = model.mixture_weight*f_c.value.*F_k.substract + (1-model.mixture_weight)*model.p_0;

    output.logs = zeros(subject_j.num_trials,1);
    output.logs(idx) = log(mixture_pdf) ;
    output.logs(~idx) = log((1-model.mixture_weight)*model.p_0);
    output.log = sum(output.logs,'omitnan');
  
    if (gradient == true)    
        output.grad = zeros(subject_j.num_trials,10);
        output.grad(idx,:) = model.mixture_weight*[f_c.grad_b.*F_k.substract F_k.grad_b.*f_c.value ...
            f_c.grad_A.*F_k.substract F_k.grad_A.*f_c.value ...
            f_c.grad_v.*F_k.substract F_k.grad_v.*f_c.value ...
            f_c.grad_s.*F_k.substract F_k.grad_s.*f_c.value ...
            f_c.grad_tau.*F_k.substract F_k.grad_tau.*f_c.value]./mixture_pdf;
    end     
end