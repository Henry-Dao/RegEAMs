%% Plot and Figure settings
clear all; clc

load('MCMC_result.mat')
t_start = 10001; % throw away draws from the burn-in and adaptation stage
t_end = size(MCMC_draws.alpha,3);

load('VB_result.mat')
lambda_best = VB_results.lambda;
mu = lambda_best.mu; B = lambda_best.B;   d = lambda_best.d;
N = 1000;
J = model.num_subjects;
D_alpha = sum(model.subject_param_dim);
num_covariates = model.beta_dim(2);
d_alpha = model.beta_dim(1);

p = D_alpha*(J + 1) + d_alpha*num_covariates;
Sigma_Psi = model.prior_par.Psi;
df = model.prior_par.v + D_alpha + J + 1;

ALPHA = zeros(D_alpha,J,N);
beta_vec = zeros(d_alpha*num_covariates,N);
mu_alpha = zeros(D_alpha,N);
vech_Sigma_alpha = zeros(D_alpha*(D_alpha+1)/2,N);
vech_C_alpha_star = zeros(D_alpha*(D_alpha+1)/2,N);
for i = 1:N
    r = VB_settings.r;
    epsilon = randn(p,1);
    z = randn(r,1);
    theta_1 = mu + B*z + d.*epsilon; % theta_1 = (alpha_1,...,alpha_J,mu_alpha,log a_1,...,log a_D)
    ALPHA(:,:,i) = reshape(theta_1(1:D_alpha*J),D_alpha,J);
    beta_vec(:,i) = theta_1(D_alpha*J + 1:(D_alpha*J+d_alpha*num_covariates));
    mu_alpha(:,i) = theta_1((D_alpha*J+d_alpha*num_covariates) + 1:end);

    Psi = Sigma_Psi;
    for j=1:J
        Psi = Psi + (ALPHA(:,j,i)-mu_alpha(:,i))*(ALPHA(:,j,i)-mu_alpha(:,i))';
    end
    Sigma_alpha = iwishrnd(Psi,df);
    vech_Sigma_alpha(:,i) = vech(Sigma_alpha);
    C = chol(Sigma_alpha,'lower');
    C_star = C; C_star(1:D_alpha+1:end) = log(diag(C));
    vech_C_alpha_star(:,i) = vech(C_star);

end

VB_results.theta_VB.alpha = ALPHA;
VB_results.theta_VB.beta_vec = beta_vec;
VB_results.theta_VB.mu_alpha = mu_alpha;
VB_results.theta_VB.vech_Sigma_alpha = vech_Sigma_alpha;
VB_results.theta_VB.vech_C_alpha_star = vech_C_alpha_star;


Big_title = '';    Big_title_size = 20;

%% Title names

num_rows = 2;   num_cols = 5;
max_subplot = num_rows*num_cols;
subplot_count = max_subplot;
fig_num = 1;
title_names ={'$\log c$','$\log A$','$v_s^s$', '$v_s^m$', ...
    '$v_m^s$', '$v_m^m$','$v_e$','$v_c$','$\log(\tau_0)$','$\log(\tau)$'};

%% Marginal Posterior Density plots

%======================= Set up legend names
legend_names = {'MCMC', 'VB'};

line_size = 1.5;
legend_xaxis = -0.08;    legend_yaxis = 0.4;
legend_sizes = 18;

D_alpha = model.subject_param_dim;
%% Fixed-effect $\beta$
d_alpha = model.beta_dim(1);    d = model.beta_dim(2);
num_rows_beta = 2;   num_cols_beta = 4;
common_legend = true;   legend_xaxis = -0.02;    legend_yaxis = 0.4;
beta_legend_xaxis = -0.06;    beta_legend_yaxis = 0.4;
max_subplot_beta = num_rows_beta*num_cols_beta;
subplot_count = max_subplot_beta;
plot_by_parameters = true;
if plot_by_parameters == true
    for i = 1:d_alpha
        subplot_count = max_subplot_beta + 1;
        for j = 1:d
            if (subplot_count > max_subplot_beta)
                figure(fig_num)
                if exist('linked_parameter_names','var') == 1
                    sgtitle({Big_title,[' Fixed-level $\beta_{vec}$ - linked parameter ',linked_parameter_names{i}]},'Fontsize',Big_title_size,'Interpreter','latex');
                else
                    sgtitle({Big_title,[' Fixed-level $\beta_{vec}$ - row ',num2str(i)]},'Fontsize',Big_title_size,'Interpreter','latex');
                end
                subplot_count = 1;
                fig_num = fig_num + 1;
            end
            subplot(num_rows_beta,num_cols_beta,subplot_count)
            hold on
            [y,x] = ksdensity(MCMC_draws.beta_vec(i+d_alpha*(j-1),t_start:t_end));
            plot(x,y,'LineWidth',line_size);
            [y,x] = ksdensity(VB_results.theta_VB.beta_vec(i+d_alpha*(j-1),:));
            plot(x,y,'LineWidth',line_size);
            if common_legend == false
                legend(legend_names,'Fontsize',legend_sizes,'Location','northeast','Interpreter',"latex");
            else
                if (subplot_count == 1)
                    % add a bit space to the figure
                    fig = gcf;
                    fig.Position(3) = fig.Position(3) + 250;
                    % add legend
                    legend(legend_names,'Fontsize',legend_sizes,'Location','northeast','Interpreter',"latex");
                    Lgnd = legend('show');
                    Lgnd.Position(1) = beta_legend_xaxis;
                    Lgnd.Position(2) = beta_legend_yaxis;
                end
            end
            title(['$\beta_{',num2str(i),',',num2str(j),'}$'],'Fontsize',18,'Interpreter',"latex")
            subplot_count = subplot_count+1;
            hold off
        end
    end
else
    max_subplot_beta = num_rows_beta*num_cols_beta;
    for j = 1:d
        subplot_count = max_subplot_beta + 1;
        for i = 1:d_alpha
            if (subplot_count > max_subplot_beta)
                figure(fig_num)
                if exist('covariate_names','var') == 1
                    sgtitle({Big_title,[' Fixed-level $\beta_{vec}$ - covariate ',covariate_names{j}]},'Fontsize',Big_title_size,'Interpreter','latex');
                else
                    sgtitle({Big_title,[' Fixed-level $\beta_{vec}$ - covariate ',num2str(j)]},'Fontsize',Big_title_size,'Interpreter','latex');
                end
                subplot_count = 1;
                fig_num = fig_num + 1;
            end
            subplot(num_rows_beta,num_cols_beta,subplot_count)
            hold on
            if MCMC == true
                for k = 1:num_MCMCs
                    if throw_away_burn_in == true
                        t_start = 1;
                    else
                        t_start = start_iter;
                    end
                    t_end = size(MCMCdraws{k,1}.beta_vec,2);
                    [y,x] = ksdensity(MCMCdraws{k,1}.beta_vec(i+d_alpha*(j-1),t_start:t_end));
                    plot(x,y,'LineWidth',line_size);
                end
            end
            if VB == true
                for k = 1:num_VBs
                    [y,x] = ksdensity(VBs{k,1}.theta_VB.beta_vec(i+d_alpha*(j-1),:));
                    plot(x,y,'LineWidth',line_size);
                end
            end
            if plot_truevalues == true
                xline(true_values.beta_matrix(i,j));
            end

            if common_legend == false
                legend(legend_names,'Fontsize',legend_sizes,'Location','northeast','Interpreter',"latex");
            else
                if (subplot_count == 1)
                    % add a bit space to the figure
                    fig = gcf;
                    fig.Position(3) = fig.Position(3) + 250;
                    % add legend
                    legend(legend_names,'Fontsize',legend_sizes,'Location','northeast','Interpreter',"latex");
                    Lgnd = legend('show');
                    Lgnd.Position(1) = beta_legend_xaxis;
                    Lgnd.Position(2) = beta_legend_yaxis;
                end
            end
            title(['$\beta_{',num2str(i),',',num2str(j),'}$'],'Fontsize',18,'Interpreter',"latex")

            hold off
            subplot_count = subplot_count+1;
        end
    end
end

%% Random effects
j = 1; % select a subject index
subplot_count = max_subplot + 1;
for i = 1:D_alpha
    if (subplot_count > max_subplot)
        figure(fig_num)
        sgtitle({Big_title,[' subject ', num2str(j)]},'FontSize',Big_title_size);
        subplot_count = 1;
        fig_num = fig_num + 1;
    end
    subplot(num_rows,num_cols,subplot_count)
    hold on

    [y,x] = ksdensity(reshape(MCMC_draws.alpha(i,j,t_start:t_end),(t_end - t_start + 1),1));
    plot(x,y,'LineWidth',line_size);
    N = size(VB_results.theta_VB.alpha,3);
    [y,x] = ksdensity(reshape(VB_results.theta_VB.alpha(i,j,:),N,1));
    plot(x,y,'LineWidth',line_size);
    title(title_names{subplot_count},'Interpreter',"latex")
    if (subplot_count == 1)
        % add a bit space to the figure
        fig = gcf;
        fig.Position(3) = fig.Position(3) + 250;
        % add legend
        legend(legend_names,'Fontsize',legend_sizes,'Location','northeast','Interpreter',"latex");
        Lgnd = legend('show');
        Lgnd.Position(1) = legend_xaxis;
        Lgnd.Position(2) = legend_yaxis;
    end
    subplot_count = subplot_count + 1;
end


%% Group-level mean $\mu_{\alpha}$

subplot_count = max_subplot + 1;
for i = 1:D_alpha
    if (subplot_count > max_subplot)
        figure(fig_num)
        sgtitle({Big_title,'Group-level mean'},'FontSize',Big_title_size);
        subplot_count = 1;
        fig_num = fig_num + 1;
    end
    subplot(num_rows,num_cols,subplot_count)
    hold on
    [y,x] = ksdensity(MCMC_draws.mu_alpha(i,t_start:t_end));
    plot(x,y,'LineWidth',line_size);
    [y,x] = ksdensity(VB_results.theta_VB.mu_alpha(i,:));
    plot(x,y,'LineWidth',line_size);
    hold off
    title(title_names{subplot_count},'Fontsize',18,'Interpreter',"latex")
    subplot_count = subplot_count+1;
end
% add a bit space to the figure
fig = gcf;
fig.Position(3) = fig.Position(3) + 250;
% add legend
legend(legend_names,'Fontsize',legend_sizes,'Location','northeast','Interpreter',"latex");
Lgnd = legend('show');
Lgnd.Position(1) = legend_xaxis;
Lgnd.Position(2) = legend_yaxis;

%% Group-level variances $\Sigma_{\alpha}$

subplot_count = max_subplot + 1;



A = reshape(1:D_alpha*D_alpha,D_alpha,D_alpha);
diagonal_index = diag(A);
A_vech = vech(A);
[~,diagonal_index_vech_A] = intersect(A_vech,diagonal_index);
for i = 1:D_alpha
    if (subplot_count > max_subplot)
        figure(fig_num)
        sgtitle({Big_title,' Group-level variances '},'FontSize',Big_title_size);
        subplot_count = 1;
        fig_num = fig_num + 1;
    end
    subplot(num_rows,num_cols,subplot_count)
    hold on

    [y,x] = ksdensity(MCMC_draws.vech_Sigma_alpha(diagonal_index_vech_A(i),t_start:t_end));
    plot(x,y,'-','LineWidth',line_size);
    [y,x] = ksdensity(VB_results.theta_VB.vech_Sigma_alpha(diagonal_index_vech_A(i),:));
    plot(x,y,'LineWidth',line_size);
    hold off
    title(title_names{subplot_count},'Fontsize',18,'Interpreter',"latex")
    if (subplot_count == 1)
        % add a bit space to the figure
        fig = gcf;
        fig.Position(3) = fig.Position(3) + 250;
        % add legend
        legend(legend_names,'Fontsize',legend_sizes,'Location','northeast','Interpreter',"latex");
        Lgnd = legend('show');
        Lgnd.Position(1) = legend_xaxis;
        Lgnd.Position(2) = legend_yaxis;
    end
    subplot_count = subplot_count+1;
end



%% Moment-plots

figure(fig_num);
baseline = MCMC_draws;
mean_baseline = [mean(baseline.beta_vec(:,t_start:t_end),2); mean(baseline.mu_alpha(:,t_start:t_end),2);...
    mean(baseline.vech_Sigma_alpha(:,t_start:t_end),2)];

min_y = min(mean_baseline);
max_y = max(mean_baseline);
y_range = max_y - min_y;
x = linspace(min_y -0.1*y_range,max_y+0.1*y_range,200);
subplot(2,2,1);
hold on;

%----------------- POSTERIOR MEANS OF GROUP-LEVEL MEAN ---------------------
mean_compare = [mean(VB_results.theta_VB.beta_vec,2);mean(VB_results.theta_VB.mu_alpha,2);...
    mean(VB_results.theta_VB.vech_Sigma_alpha,2)];
plot(mean_compare,mean_baseline,'o','MarkerFaceColor',[0 0.4470 0.7410],'LineWidth',2.5);
set(gca,'fontsize',18) % previously 18, but no title
xlabel('Mean estimated by VB');
ylabel('Mean estimated by PMwG');
title('posterior mean (group parameter)');
plot(x,x,'r--','LineWidth',1,'HandleVisibility','off');
hold off;

%----------------- POSTERIOR STD OF GROUP-LEVEL MEAN ---------------------

std_baseline = [std(baseline.beta_vec(:,t_start:t_end),0,2);...
    std(baseline.mu_alpha(:,t_start:t_end),0,2);...
    std(baseline.vech_Sigma_alpha(:,t_start:t_end),0,2)];
min_y = min(std_baseline);
max_y = max(std_baseline);
y_range = max_y - min_y;
x = linspace(min_y -0.1*y_range,max_y+0.1*y_range,200);
subplot(2,2,2);
hold on;

%----------------- POSTERIOR MEAN PLOT OF mu_alpha ---------------------
std_compare = [std(VB_results.theta_VB.beta_vec,0,2);...
    std(VB_results.theta_VB.mu_alpha,0,2);...
    std(VB_results.theta_VB.vech_Sigma_alpha,0,2)];
plot(std_compare,std_baseline,'o','MarkerFaceColor',[0 0.4470 0.7410],'LineWidth',2.5);
set(gca,'fontsize',18) % previously 18, but no title
xlabel('Mean estimated by VB');
ylabel('Mean estimated by PMwG');
title('posterior standard deviation (group parameter)');
plot(x,x,'r--','LineWidth',1,'HandleVisibility','off');
hold off;

%----------------- POSTERIOR MEAN PLOT OF RANDOM EFFECTS ---------------------
mean_alpha_baseline = mean(baseline.alpha(:,:,t_start:t_end),3);
min_y = min(mean_alpha_baseline(:));
max_y = max(mean_alpha_baseline(:));
y_range = max_y - min_y;
x = linspace(min_y -0.1*y_range,max_y+0.1*y_range,200);
subplot(2,2,3);
hold on;

mean_alpha_compare = mean(VB_results.theta_VB.alpha,3);
plot(mean_alpha_compare(:),mean_alpha_baseline(:),'o','MarkerFaceColor',[0 0.4470 0.7410],'LineWidth',2.5);
set(gca,'fontsize',18) % previously 18, but no title
xlabel('Mean estimated by VB');
ylabel('Mean estimated by PMwG');
title('posterior mean (subject parameters)');
plot(x,x,'r--','LineWidth',1,'HandleVisibility','off');
hold off;


%----------------- POSTERIOR STD OF RANDOM EFFECTS ---------------------
std_alpha_baseline = std(baseline.alpha(:,:,t_start:t_end),0,3);
min_y = min(std_alpha_baseline(:));
max_y = max(std_alpha_baseline(:));
y_range = max_y - min_y;
x = linspace(min_y -0.1*y_range,max_y+0.1*y_range,200);
subplot(2,2,4);
hold on;
std_alpha_compare = std(VB_results.theta_VB.alpha,0,3);
plot(std_alpha_compare(:),std_alpha_baseline(:),'o','MarkerFaceColor',[0 0.4470 0.7410],'LineWidth',2.5);

set(gca,'fontsize',18)
xlabel('Std estimated by VB');
ylabel('Std estimated by PMwG');
title('Posterior standard deviation (subject parameters)');
plot(x,x,'r--','LineWidth',1,'HandleVisibility','off');
hold off;



