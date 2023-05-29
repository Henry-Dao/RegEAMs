function scaled_data = scale_data(data,scaled_std)
J = length(data);
X = [];
for j =1:J
    X = [X; data{j,1}.X];
end
for j =1:J
    data{j,1}.X = (data{j,1}.X - mean(X,1))./(std(X)/scaled_std);
end
scaled_data = data;
end

%% Test: copy and run the code below in a new script file
% computer_path = 'C:/Users/z5174294/'; % 'E:/';
% main_path = 'Dropbox/ACADEMIC_CAREER/STATISTICAL_MODELS/Hierarchical_Regression_Models/';
% data_path = [computer_path, main_path, 'Data_and_Results/'];
% data_name = 'Gaussian_Reg_simdata_exp3';
% load([data_path,data_name,'.mat']) % load the data
% 
% scaled_data = scale_data(data,0.1);
% J = length(data);
% for j =1:J
%     mean(scaled_data{j,1}.X,1)
%     std(scaled_data{j,1}.X,1)
% end