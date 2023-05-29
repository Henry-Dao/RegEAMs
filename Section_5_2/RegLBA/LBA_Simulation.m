function [choice, RT] = LBA_Simulation(A, b, v, t0, sv,silent)

%   1. Input are matrices of size n by N 
%         - N is the number of accumulators.
%         - n is the number of trials
% Usage: [choice RT conf] = LBA_trial(A, b, v, t0, sv, N)

% Outputs:
%
% choice = scalar from 1:N indicating response chosen by model
% RT = reaction time in ms


[n, N] = size(A); 
% n = number of simulated trials; N = number of choices (accumulators)

k = zeros(n,N);     d = zeros(n,N);     t = zeros(n,N);
allRT = zeros(n,N);
RT = zeros(n,1);    choice = zeros(n,1);

for i = 1:n
    trialOK = false;
    if silent == false % && mod(i,ceil(0.25*n)) == 0 
        disp([' Simulated data: ',num2str(round(100*i/n,0)),' % completed !'])
    end
    count_nonresponses = 1;
    while ~trialOK
        for j = 1:N
            
            % Get starting point
            k(i,j) = rand().*A(i,j);
            
            % Get drift rate
            d(i,j) = normrnd(v(i,j), sv(i,j));
            
            % Get time to threshold
            t(i,j) = (b(i,j)-k(i,j))./d(i,j);
            
            % Add on non-decision time
            allRT(i,j) = t0(i,j) + t(i,j);
        end
        non_responses_idx = d(i,:)<0;
        if sum(non_responses_idx) == N          
            RT(i) = inf;
            choice(i) = -1;
            trialOK = count_nonresponses >100;
            count_nonresponses = count_nonresponses + 1;        
        else
            % Get choice and confidence
            allRT(i,non_responses_idx) = inf; 
            [RT(i), choice(i)] = min(allRT(i,:));
            trialOK = true;
        end
    end
end