function alpha_j_R_filtered = filtering_alpha(alpha_j_R,lower_threshold,upper_threshold)
    idx = (min(alpha_j_R,[],2) > lower_threshold)&(max(alpha_j_R,[],2) < upper_threshold);
    alpha_j_R_filtered = alpha_j_R(idx,:);
end