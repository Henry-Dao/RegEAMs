%% Settings
clear all; clc
%% ----------------- Boxplot Summary Statistics --------------
T_pred = 20;
result_names = cell(2,1);
result_names{1,1} = 'MCMC';
result_names{2,1} = 'VB';

legend_names = {'Data','PMwG','VB'};

Big_title = 'Mental Rotation data';    Big_title_size = 20;
%%  Observed data
load('ERP_data.mat')
scaling_data = true;  scaled_std = 1;
if scaling_data == true
    data = scale_data(data,scaled_std);
end
%---------------- Mean Proportion Correct --------------------
cond_name = {' 0 ',' 45 ',' 90 ',' 135 ',' 180 '};
J = length(data);
P_c.same = zeros(5,J);   P_c.mirror = zeros(5,J);
for j = 1:J
    S_j = data{j,1}.S;
    E_j = data{j,1}.E;
    n_j = data{j,1}.num_trials;
    SE_j = zeros(n_j,5);
    for e = 1:5
        P_c.same(e,j) = sum((data{j,1}.R == S_j).*(S_j ==1).*(E_j == e))/sum((S_j ==1).*(E_j == e));
        P_c.mirror(e,j) = sum((data{j,1}.R == S_j).*(S_j ==2).*(E_j == e))/sum((S_j ==2).*(E_j == e));
    end
end
Mean_P_c.same = zeros(5,1);     Mean_P_c.mirror = zeros(5,1);
for e = 1:5
    Mean_P_c.same(e,1) = mean(P_c.same(e,:));
    disp([' Mean correct proportion S = same and E = ',cond_name{e},' is ',num2str(Mean_P_c.same(e,1))])
end
for e = 1 :5
    Mean_P_c.mirror(e,1) = mean(P_c.mirror(e,:));
    disp([' Mean correct proportion S = mirror and E = ',cond_name{e},' is ',num2str(mean(Mean_P_c.mirror(e,1)))])
end
%---------------- Median RT --------------------
J = length(data);
Median_RT.same = zeros(5,J);   Median_RT.mirror = zeros(5,J);
for j = 1:J
    S_j = data{j,1}.S;
    E_j = data{j,1}.E;
    n_j = data{j,1}.num_trials;
    SE_j = zeros(n_j,5);
    for e = 1:5
        idx = (S_j == 1)&(E_j == e);
        Median_RT.same(e,j) = median(data{j,1}.RT(idx));

        idx = (S_j == 2)&(E_j == e);
        Median_RT.mirror(e,j) = median(data{j,1}.RT(idx));
    end
end
Mean_Median_RT.same = zeros(5,1);     Mean_Median_RT.mirror = zeros(5,1);
for e =1 :5
    Mean_Median_RT.same(e,1) = mean(Median_RT.same(e,:));
    disp([' Mean Median RT when S = same and E = ',cond_name{e},' is ',num2str(Mean_Median_RT.same(e,1))])
end
for e =1 :5
    Mean_Median_RT.mirror(e,1) = mean(Median_RT.mirror(e,:));
    disp([' Mean Median RT when S = mirror and E = ',cond_name{e},' is ',num2str(Mean_Median_RT.mirror(e,1))])
end

%% ----------------- Generate Post predictive data --------------
K = length(result_names);
Mean_Pred_P_c = cell(K,1);  Mean_Pred_RT = cell(K,1);
for k = 1: K
    
    %----------- Generate posterior predictive distribution
    post_pred_data = cell(T_pred,1);

    for j = 1:J
        disp(['---------- Simulating subject ',num2str(j),'------------'])
        load([result_names{k,1},'_R_to_Matlab_predicted_data_subject_',num2str(j),'.mat'])
        for i = 1:T_pred
            n_j = data{j,1}.num_trials;
            post_pred_data{i,1}.data{j,1}.num_trials = n_j;
            post_pred_data{i,1}.data{j,1}.X = data{j,1}.X;
            post_pred_data{i,1}.data{j,1}.S = data{j,1}.S;
            post_pred_data{i,1}.data{j,1}.E = data{j,1}.E;
            post_pred_data{i,1}.data{j,1}.RT = predicted_data.rt(((i-1)*n_j+1):(i*n_j));
            post_pred_data{i,1}.data{j,1}.R = predicted_data.response(((i-1)*n_j+1):(i*n_j));
        end
        clear predicted_data
    end

    J = length(post_pred_data{1,1}.data);
    Mean_Pred_P_c{k,1}.same = zeros(5,T_pred);     Mean_Pred_P_c{k,1}.mirror = zeros(5,T_pred);
    Mean_Pred_RT{k,1}.same = zeros(5,T_pred);     Mean_Pred_RT{k,1}.mirror = zeros(5,T_pred);
    for t = 1:T_pred
        Pred_P_c_j.same = zeros(5,J);   Pred_P_c_j.mirror = zeros(5,J);
        Pred_RT_j.same = zeros(5,J);   Pred_RT_j.mirror = zeros(5,J);
        for e = 1:5
            for j = 1:J
                nonresponses_idx = post_pred_data{t,1}.data{j,1}.RT == inf;
                S_j = post_pred_data{t,1}.data{j,1}.S(~nonresponses_idx);
                E_j = post_pred_data{t,1}.data{j,1}.E(~nonresponses_idx);
                RE_j = post_pred_data{t,1}.data{j,1}.R(~nonresponses_idx);
                RT_j = post_pred_data{t,1}.data{j,1}.RT(~nonresponses_idx);
                n_j = sum(~nonresponses_idx);
                if n_j == 0
                    disp('all nonresponses')
                end
                SE_j = zeros(n_j,5);
                Pred_P_c_j.same(e,j) = sum((RE_j == S_j).*(S_j ==1).*(E_j == e))/sum((S_j ==1).*(E_j == e));
                Pred_P_c_j.mirror(e,j) = sum((RE_j == S_j).*(S_j ==2).*(E_j == e))/sum((S_j ==2).*(E_j == e));

                idx = (S_j == 1)&(E_j == e);
                Pred_RT_j.same(e,j) = median(post_pred_data{t,1}.data{j,1}.RT(idx));

                idx = (S_j == 2)&(E_j == e);
                Pred_RT_j.mirror(e,j) = median(post_pred_data{t,1}.data{j,1}.RT(idx));
            end
            Mean_Pred_P_c{k,1}.same(e,t) = mean(Pred_P_c_j.same(e,:));
            Mean_Pred_P_c{k,1}.mirror(e,t) = mean(Pred_P_c_j.mirror(e,:));

            Mean_Pred_RT{k,1}.same(e,t) = mean(Pred_RT_j.same(e,:));
            Mean_Pred_RT{k,1}.mirror(e,t) = mean(Pred_RT_j.mirror(e,:));
        end
    end
    clear post_pred_data;
end

%% ----------------- Create predictive plots --------------
error_bar_tail_size = 10;
error_bar_size = 2;

figure(1)
% ----------------- Mean accuracy (same) -----------------
subplot(2,2,1)
hold on

x = 1:1:5;
data_points = Mean_P_c.same;
percentiles = prctile(Mean_Pred_P_c{1,1}.same,[25 50 75],2);
y1 = percentiles(:,2);
yneg1 = abs(y1 - percentiles(:,1));
ypos1 = abs(y1 - percentiles(:,3));

percentiles = prctile(Mean_Pred_P_c{2,1}.same,[25 50 75],2);
y2 = percentiles(:,2);
yneg2 = abs(y2 - percentiles(:,1));
ypos2 = abs(y2 - percentiles(:,3));

% plot data points and lines
plot(x,data_points,'.-','LineWidth',1,'MarkerSize',36)

% plot first error bars
e1 = errorbar(x-0.15,y1,yneg1,ypos1,'.','MarkerSize',27,'CapSize',error_bar_tail_size);
e1.LineWidth = error_bar_size;

% plot second error bars
e2 = errorbar(x+0.15,y2,yneg2,ypos2,'.','MarkerSize',27,'CapSize',error_bar_tail_size);
e2.LineWidth = error_bar_size;

hold off

% set axis limits and labels
ylim([0.88 1])
ylabel('Mean accuracy','FontSize',16)
title('Same','FontSize',16)

% set custom tick labels for x-axis
xticks(x)
xticklabels({'0\circ','45\circ','90\circ','135\circ','180\circ','label 4','label 4'})

% set legend
% legend('Group 1','Group 2','Location','northwest')

% ------- to change axis font size (I add) ---------
ax = gca;
ax.FontSize = 22;

% ----------------- Mean accuracy (mirror) -----------------
subplot(2,2,2)
hold on

x = 1:1:5;
data_points = Mean_P_c.mirror;
percentiles = prctile(Mean_Pred_P_c{1,1}.mirror,[25 50 75],2);
y1 = percentiles(:,2);
yneg1 = abs(y1 - percentiles(:,1));
ypos1 = abs(y1 - percentiles(:,3));

percentiles = prctile(Mean_Pred_P_c{2,1}.mirror,[25 50 75],2);
y2 = percentiles(:,2);
yneg2 = abs(y2 - percentiles(:,1));
ypos2 = abs(y2 - percentiles(:,3));

% plot data points and lines
plot(x,data_points,'.-','LineWidth',1,'MarkerSize',36)

% plot first error bars
e1 = errorbar(x-0.15,y1,yneg1,ypos1,'.','MarkerSize',27,'CapSize',error_bar_tail_size);
e1.LineWidth = error_bar_size;

% plot second error bars
e2 = errorbar(x+0.15,y2,yneg2,ypos2,'.','MarkerSize',27,'CapSize',error_bar_tail_size);
e2.LineWidth = error_bar_size;

hold off

ylim([0.88 1])

title('Mirror','FontSize',16)

% set custom tick labels for x-axis
xticks(x)
xticklabels({'0\circ','45\circ','90\circ','135\circ','180\circ','label 4','label 4'})


% ------- to change axis font size (I add) ---------
ax = gca;
ax.FontSize = 22;


% ----------------- Mean Median RT (same) -----------------
subplot(2,2,3)
hold on

x = 1:1:5;
data_points = Mean_Median_RT.same;
percentiles = prctile(Mean_Pred_RT{1,1}.same,[25 50 75],2);
y1 = percentiles(:,2);
yneg1 = abs(y1 - percentiles(:,1));
ypos1 = abs(y1 - percentiles(:,3));

percentiles = prctile(Mean_Pred_RT{2,1}.same,[25 50 75],2);
y2 = percentiles(:,2);
yneg2 = abs(y2 - percentiles(:,1));
ypos2 = abs(y2 - percentiles(:,3));

% plot data points and lines
plot(x,data_points,'.-','LineWidth',1,'MarkerSize',36)

% plot first error bars
e1 = errorbar(x-0.15,y1,yneg1,ypos1,'.','MarkerSize',27,'CapSize',error_bar_tail_size);
e1.LineWidth = error_bar_size;

% plot second error bars
e2 = errorbar(x+0.15,y2,yneg2,ypos2,'.','MarkerSize',27,'CapSize',error_bar_tail_size);
e2.LineWidth = error_bar_size;

hold off

% set axis limits and labels
ylim([0.6 1.4])
xlabel('Rotation Angle','FontSize',16) % I add this
ylabel('Median RT','FontSize',16)

% set custom tick labels for x-axis
xticks(x)
xticklabels({'0\circ','45\circ','90\circ','135\circ','180\circ','label 4','label 4'})

% set legend
% legend('Group 1','Group 2','Location','northwest')

% ------- to change axis font size (I add) ---------
ax = gca;
ax.FontSize = 22;

% ----------------- Mean Median RT (mirror) -----------------
subplot(2,2,4)
hold on

x = 1:1:5;
data_points = Mean_Median_RT.mirror;
percentiles = prctile(Mean_Pred_RT{1,1}.mirror,[25 50 75],2);
y1 = percentiles(:,2);
yneg1 = abs(y1 - percentiles(:,1));
ypos1 = abs(y1 - percentiles(:,3));

percentiles = prctile(Mean_Pred_RT{2,1}.mirror,[25 50 75],2);
y2 = percentiles(:,2);
yneg2 = abs(y2 - percentiles(:,1));
ypos2 = abs(y2 - percentiles(:,3));

% plot data points and lines
plot(x,data_points,'.-','LineWidth',1,'MarkerSize',36)

% plot first error bars
e1 = errorbar(x-0.15,y1,yneg1,ypos1,'.','MarkerSize',27,'CapSize',error_bar_tail_size);
e1.LineWidth = error_bar_size;

% plot second error bars
e2 = errorbar(x+0.15,y2,yneg2,ypos2,'.','MarkerSize',27,'CapSize',error_bar_tail_size);
e2.LineWidth = error_bar_size;

hold off

% set axis limits and labels
% xlim([0.5 6])
ylim([0.6 1.4])
xlabel('Rotation Angle','FontSize',16) 

% set custom tick labels for x-axis
xticks(x)
xticklabels({'0\circ','45\circ','90\circ','135\circ','180\circ','label 4','label 4'})


% ------- to change axis font size (I add) ---------
ax = gca;
ax.FontSize = 22;

