function scaled_data = scale_data(data,scaled_std)
J = length(data);

for j =1:J
    data{j,1}.X = (data{j,1}.X - mean(data{j,1}.X,1))./(std(data{j,1}.X)/scaled_std);
end
scaled_data = data;
end
