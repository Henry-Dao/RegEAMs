function Y_stacked = RT_Stacking_observations(subject_j,R)
    subject_j.RT = repmat(subject_j.RT,R,1);
    subject_j.R = repmat(subject_j.R,R,1);
    subject_j.S = repmat(subject_j.S,R,1);
    subject_j.E = repmat(subject_j.E,R,1);
    subject_j.num_trials = subject_j.num_trials*R;
    Y_stacked = subject_j;
end