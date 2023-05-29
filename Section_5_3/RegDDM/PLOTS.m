%% Description: create the plot in Section 5.3 - HCP

clear all;  clc

Big_title = 'RegDDM';    Big_title_size = 20;
load('VB_result.mat')

lambda_best = VB_results.lambda;
mu = lambda_best.mu; B = lambda_best.B;   d = lambda_best.d;
N = 5000;

r = VB_settings.r;
J = model.num_subjects;
D_alpha = sum(model.subject_param_dim);
num_covariates = model.beta_dim(2);
d_alpha = model.beta_dim(1);

p = D_alpha*(J + 1) + d_alpha*num_covariates;
Sigma_Psi = model.prior_par.Psi;
df = model.prior_par.v + D_alpha + J + 1;

ALPHA = zeros(D_alpha,J,N);
beta_vec = zeros(d_alpha*num_covariates,N);

for i = 1:N
    for j=1:J
        epsilon = randn(D_alpha,1);
        z = randn(r,1);
        theta_temp = mu{j,1} + B{j,1}*z + d{j,1}.*epsilon;
        ALPHA(:,j,i) = theta_temp;
    end
    epsilon=randn(D_alpha+d_alpha*num_covariates,1);
    z=randn(r,1);
    beta_vec_mu_alpha=mu{J+1,1}+B{J+1,1}*z + d{J+1,1}.*epsilon;
    beta_vec(:,i) = beta_vec_mu_alpha(1 : d_alpha*num_covariates);
end
VB_results.theta_VB.alpha = ALPHA;
VB_results.theta_VB.beta_vec = beta_vec;

linked_parameter_simplified_names ={'$\textrm{lure}$','$\textrm{target}$','$\textrm{nontarget}$','$\textrm{boundary}$'};
covariate_names ={'MMSE','PSQI','Sequencing', 'Card Sort','Vocab','Speed','Fluid', 'Crystal'}; % optional


error_bar_tail_size = 10;
error_bar_size = 2;
fig_count = 9;
fig_num = 1;
for i = 1:model.beta_dim(2)
    if fig_count >8
        figure(fig_num);
        fig_num = fig_num + 1;
        fig_count = 1;
    end
    subplot(2,4,fig_count)
    fig_count = fig_count + 1;
    idx = ((i-1)*model.beta_dim(1) + 1):(i*model.beta_dim(1));

    hold on
    x = 1:1:4;
    percentiles = prctile(VB_results.theta_VB.beta_vec(idx(1:4),:),[2.5 50 97.5],2);
    y1 = percentiles(:,2);
    yneg1 = abs(y1 - percentiles(:,1));
    ypos1 = abs(y1 - percentiles(:,3));

    percentiles = prctile(VB_results.theta_VB.beta_vec(idx(5:8),:),[2.5 50 97.5],2);
    y2 = percentiles(:,2);
    yneg2 = abs(y2 - percentiles(:,1));
    ypos2 = abs(y2 - percentiles(:,3));

    % plot first error bars
    e1 = errorbar(x-0.15,y1,yneg1,ypos1,'.','MarkerSize',27,'CapSize',error_bar_tail_size);
    e1.LineWidth = error_bar_size;

    % plot second error bars
    e2 = errorbar(x+0.15,y2,yneg2,ypos2,'.','MarkerSize',27,'CapSize',error_bar_tail_size);
    e2.LineWidth = error_bar_size;


    % set axis limits and labels
    xlim([0.5 4.5])
    ylim([-0.1 0.28])

    yline(0,'LineWidth',1)
    xaxisproperties= get(gca, 'XAxis');
    xaxisproperties.TickLabelInterpreter = 'latex'; % latex for x-axis

    xticks(x)
    xticklabels(linked_parameter_simplified_names)
    ax = gca;
    ax.FontSize = 18;
    title(covariate_names(i),'FontSize',22,'Interpreter','latex')
    hold off
end
